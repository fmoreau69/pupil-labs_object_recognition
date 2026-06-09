# Roadmap — Pupil Labs Object Recognition (refonte 2026)

Plugin de reconnaissance des objets observés pour **Pupil Capture / Pupil Player**, dans le
cadre d'expérimentations en sciences cognitives (conduite sur simulateur ou véhicule
expérimental). Objectif : à partir de la vidéo de la caméra de scène d'un casque **Pupil Core**,
détecter/segmenter les objets d'intérêt et déterminer **ce que regarde le participant** (et pas
seulement *où* il regarde), en temps réel, avec enregistrement et diffusion des données.

Refonte 2026 : abandon de **Darknet** et **imagezmq** au profit d'**Ultralytics (YOLO)** et de
**pyzmq** nu.

---

## Contrainte structurante : Python du bundle

Le bundle officiel **Pupil Capture v3.5.1** embarque **Python 3.6** (`python36.dll`). Ultralytics
et PyTorch n'ont pas de wheels pour 3.6 → **impossible de faire tourner YOLO « in-process »** dans
le plugin du bundle. Faire tourner Pupil Capture depuis les sources sur un Python moderne est
possible mais lourd sur Windows (build de `pupil-detectors`, `pyglui`, `pyav`…), à l'opposé de
l'objectif « installation simple ».

**→ Décision : architecture en deux processus.**

```
┌─ Pupil Capture (bundle, Python 3.6) ───────────────┐
│  detection_plugin.py  — MINCE, 0 dépendance lourde  │
│   • recent_events : world frame + gaze              │
│   • envoie la frame ──ZMQ──> détecteur              │
│   • reçoit boxes / masques / labels <──ZMQ          │
│   • matching gaze↔objet (rouge = observé / vert)    │
│   • overlay (cv2 + gl_display)                      │
│   • publie "objects" sur l'IPC backbone Pupil       │
│   • écrit les données dans le dossier de recording  │
│  deps déjà présentes : pyzmq, numpy, opencv, msgpack│
└─────────────────────────────────────────────────────┘
         │  ZMQ localhost (frame ↓ / détections ↑)
         ▼
┌─ yolo_server.py (venv Python 3.12) ────────────────┐
│   • ultralytics YOLO(...).predict(frame)            │
│   • renvoie détections + masques (msgpack)          │
│   • (phases ult.) pylsl outlet / vidéo annotée      │
│  deps : ultralytics, torch(CUDA), pyzmq, msgpack    │
└─────────────────────────────────────────────────────┘
         │  (phases ultérieures)
         ▼  RTMaps (ZMQ), LabRecorder / WAMA (LSL)
```

**Avantages** : installation du plugin = déposer **un seul `.py`** dans
`~/pupil_capture_settings/plugins/` ; les ~2 Go de PyTorch restent dans un venv que l'on maîtrise ;
temps réel préservé (ZMQ localhost < 1 ms) ; `imagezmq` supprimé.

---

## Environnements

| Environnement              | Python | Rôle                                              |
|----------------------------|:------:|---------------------------------------------------|
| Bundle Pupil Capture       |  3.6   | imposé par le `.msi` — plugin mince uniquement    |
| Venv détecteur (`.venv`)   | **3.12** | ultralytics + torch CUDA + serveur ZMQ          |

Venv validé sur la machine de dev : `torch 2.12.0+cu126`, CUDA **True**,
GPU **RTX 4070 Ti SUPER 16 Go**, `ultralytics 8.4.60`.

---

## Modèles & moteurs

Le serveur expose un système de **moteurs** (`engines.py`) combinables via `--engines` :

| Moteur    | Sortie                                   | Temps réel | Dépendances                          |
|-----------|------------------------------------------|:----------:|--------------------------------------|
| `yolo`    | objets (instances) + masques, tracking   |    oui     | ultralytics (présent)                |
| `yolopv2` | couches *sémantiques* route + voies      |    oui     | `yolopv2.pt` (TorchScript)           |
| `sam3`    | masques par prompt texte (markings/route)| **non** *  | `sam3` + HF `facebook/sam3` (token)  |

\* SAM3 sur Windows tourne frame-by-frame (le video predictor exige `triton`/Linux) → réservé au
post-traitement Pupil Player, ou à très basse fréquence en Capture.

- **Temps réel (Capture)** : `yolo11n-seg.pt` (défaut) ou `yolo11n.pt` (détection seule, PC
  faible). Combinable avec `yolopv2` : `--engines yolo,yolopv2` → objets **+** route/voies en un
  seul overlay.
- **Hors-ligne (Player)** : `sam3` (segmentation riche par prompt). Wrappers adaptés du projet WAMA
  `cam_analyzer`.
- Le plugin distingue **objets** (instances, éligibles au regard, rouge/vert) et **couches**
  (sémantiques, remplissage translucide, non éligibles au focus).

### Lancer le serveur
```powershell
python detector/yolo_server.py                           # yolo seg (défaut)
python detector/yolo_server.py --engines yolo,yolopv2 --yolopv2-model models/yolopv2.pt
python detector/yolo_server.py --engines sam3 --sam3-road "drivable road surface in front of the vehicle"
```

---

## Transports de données

| Donnée                                   | Transport                         | Destination            | Phase |
|------------------------------------------|-----------------------------------|------------------------|:-----:|
| Objets (label, box/masque, conf, gaze)   | IPC backbone + `.pldata` + `.csv` | Pupil natif / Player / analyse |  v1   |
| Objets                                   | **pylsl** outlet                  | LabRecorder / WAMA     |   3   |
| Objets                                   | **pyzmq** PUB                     | RTMaps                 |   1+  |
| Vidéo world brute                        | (enregistrée par Pupil)           | —                      |   —   |
| Vidéo annotée (overlay)                  | pyzmq PUB (JPEG) + `world_overlay.mp4` | RTMaps / fichier  | fait (5) |

**Vidéo + LSL** : on ne pousse pas la vidéo dans LSL (pertes de frames). La vidéo brute est
enregistrée par Pupil ; l'overlay est **reconstruit à la relecture** dans le plugin Pupil Player à
partir des données objets horodatées.

---

## Phases

### Phase 0 — Socle (fait)
- Venv 3.12 + torch CUDA + ultralytics validés.
- `yolo_server.py` minimal : reçoit une frame ZMQ → renvoie boxes/masques.
- Plugin mince : envoie la frame, affiche les détections.
- **Critère** : détection/segmentation temps réel de bout en bout dans Pupil Capture.

### Phase 1 — Parité 2020, propre
- Matching gaze↔objet (point-dans-boîte **et** point-dans-masque), code couleur rouge/vert.
- Publication propre sur l'IPC backbone (`g_pool.ipc_pub.send`, topic `objects`).
- Écriture des données dans le dossier de recording (`file_methods.PLData_Writer`) → re-chargeable
  dans Pupil Player.
- **Export ZMQ PUB → RTMaps** — **fait** : le plugin publie le datum par frame sur un socket PUB
  (switch « Stream object data (RTMaps/LSL) », bind `tcp://*:5561`). `rtmaps_stream.py` (réécrit en
  pyzmq nu, `imagezmq` supprimé) le consomme en SUB et expose : objet observé, boîte, id, gaze,
  nombre d'objets, timestamp Pupil, JSON complet.

### Phase 2 — Segmentation (intégrée dès la v1)
- `yolo11n-seg`, overlay des masques, matching point-dans-polygone (`cv2.pointPolygonTest`).
- Gain qualitatif : « regarde la voiture » vs « regarde le ciel derrière la boîte ».

### Phase 3 — LSL — **fait**
- `lsl_relay.py` (venv 3.12) : SUB sur le socket PUB du plugin → deux outlets LSL horodatés
  `local_clock()` : numérique 7 canaux `[observed, x1,y1,x2,y2, gaze_x, gaze_y]` + flux marqueur
  string (datum JSON complet). `pylsl` reste hors du bundle 3.6 (`requirements-relay.txt`).
- Base de la synchro multi-sources pour l'intégration WAMA (gaze + véhicule + EEG/GSR…).
- ⚠️ À valider en live : sortie réelle vers LabRecorder (pylsl non installé/testé ici).

### Phase 4 — Pupil Player + moteurs avancés
- **Moteurs serveur** (`engines.py`) : abstraction multi-moteurs `yolo` / `yolopv2` / `sam3`,
  combinables, lissage généralisé (instances par id, couches sémantiques par nom). **Fait** côté
  serveur (yolopv2/sam3 adaptés des wrappers WAMA, à valider sur machine équipée).
- **Plugin Pupil Player** (`player_object_recognition.py`) — **fait** (à valider en live), deux modes :
  1. **Relecture** : si `objects.pldata` existe dans le recording, recharge les données et
     reconstruit l'overlay sur la vidéo enregistrée (aucune ré-inférence).
  2. **Retraitement offline** : pour des recordings **bruts non traités** (world video + `gaze`,
     sans `objects.pldata`), itère sur les frames (lecture directe `world.mp4` +
     `world_timestamps.npy` + `gaze.pldata`), ré-applique la détection (client ZMQ vers le serveur,
     mêmes moteurs) **et le matching du regard** (plus proche gaze par timestamp), puis écrit
     `objects.pldata` (masques complets pour tous les objets → relecture fidèle) **et `objects.csv`**
     (objet observé à plat, pour fusion offline par timestamp avec les logs LSL/XDF + RTMaps).
     Traite a posteriori des données oculo déjà acquises sans le plugin. Idéal pour SAM3 hors temps réel.
- Le matching gaze réutilise la même logique point-dans-boîte / point-dans-masque que Capture.
- ⚠️ À vérifier en live : rendu Player (`frame.img` vs `gl_display`) et API
  `PLData_Writer`/`load_pldata_file` du bundle (codés selon l'API standard, non exécutés ici).

### Phase 4bis — Sélection des classes (fait, Capture)
- Filtre d'affichage par classe dans le plugin (`classes_filter`, liste blanche temps réel) : YOLO
  détecte tout (aucun impact perf), **toutes** les détections restent enregistrées, seul l'overlay
  et l'éligibilité au focus sont filtrés. Liste des classes vues affichée pour la découverte.

### Phase 5 — Vidéo annotée — **fait**
- **Stream** : socket ZMQ PUB séparé (switch « Stream annotated video (RTMaps) », bind `tcp://*:5562`)
  qui pousse la frame surchargée en JPEG (`[b"frame", jpg]`). `rtmaps_video.py` (pyzmq nu) la reçoit
  et l'expose en IPL_IMAGE dans RTMaps.
- **Enregistrement** : switch « Record annotated video » → écrit `world_overlay.mp4` (codec mp4v)
  dans le dossier du recording pendant un enregistrement Pupil. Repli propre + warning si le codec
  manque. Les deux ne tournent que lorsque la détection est active.
- ⚠️ À valider en live : réception RTMaps (`rtmaps_video.py`) et codec mp4v dans l'OpenCV du bundle.

---

## Découpage des fichiers

| Fichier                              | Env    | Rôle                                                  |
|--------------------------------------|:------:|-------------------------------------------------------|
| `detector/yolo_server.py` + `engines.py` | 3.12 | Serveur ZMQ REP multi-moteurs : frame → détections/masques |
| `plugins/detection_plugin.py`        | 3.6    | Plugin Pupil Capture mince (client ZMQ, gaze, overlay, IPC, export PUB) |
| `plugins/player_object_recognition.py` | 3.6  | Plugin Pupil Player (relecture + retraitement offline)|
| `integrations/rtmaps_stream.py`      | RTMaps | Réception ZMQ côté RTMaps (pyzmq nu) → données objet   |
| `integrations/rtmaps_video.py`       | RTMaps | Réception ZMQ côté RTMaps → vidéo annotée (IPL_IMAGE)  |
| `integrations/lsl_relay.py`          | 3.12   | Relais ZMQ→LSL (outlets numérique + JSON) pour synchro multi-capteurs |
| `integrations/RTMaps/`               | RTMaps | Diagrammes d'acquisition RTMaps d'exemple (`.rtd`)    |
| `detector/requirements-detector.txt` / `integrations/requirements-relay.txt` | 3.12 | Dépendances venv détecteur / relais LSL |
| `models/`                            | —      | Cache des poids modèles téléchargés (git-ignored)     |
| `~Archives/`                         | —      | `darknet`, `imagezmq`, ancien plugin (nettoyage versionning ultérieur) |

---

## Installation (cible)

1. **Détecteur** (une fois, dans le venv 3.12) :
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r detector/requirements-detector.txt
   ```
2. **Plugins** :
   - Capture : copier `plugins/detection_plugin.py` dans `~/pupil_capture_settings/plugins/`.
   - Player : copier `plugins/player_object_recognition.py` dans `~/pupil_player_settings/plugins/`.
3. **Lancement** : démarrer `python detector/yolo_server.py` puis Pupil Capture/Player ; activer le
   plugin « Object Recognition (YOLO) » dans le Plugin Manager.
   - Player : si le recording contient déjà `objects.pldata` → relecture auto de l'overlay ; sinon
     bouton **Reprocess recording** (détecteur lancé) pour le générer.
4. **Export multi-capteurs** (optionnel) : dans le plugin Capture, activer « Stream object data
   (RTMaps/LSL) ». Puis :
   - **RTMaps** : `integrations/rtmaps_stream.py` dans un bloc Python RTMaps (propriété `sub_address`).
   - **LSL** : `pip install -r integrations/requirements-relay.txt` puis
     `python integrations/lsl_relay.py` (`--connect tcp://<hôte_pupil>:5561`) ; les flux apparaissent
     dans LabRecorder.
