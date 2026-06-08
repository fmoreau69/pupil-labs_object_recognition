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
python yolo_server.py                                    # yolo seg (défaut)
python yolo_server.py --engines yolo,yolopv2 --yolopv2-model path\to\yolopv2.pt
python yolo_server.py --engines sam3 --sam3-road "drivable road surface in front of the vehicle"
```

---

## Transports de données

| Donnée                                   | Transport                         | Destination            | Phase |
|------------------------------------------|-----------------------------------|------------------------|:-----:|
| Objets (label, box/masque, conf, gaze)   | IPC backbone Pupil + `.pldata`    | Pupil natif / Player   |  v1   |
| Objets                                   | **pylsl** outlet                  | LabRecorder / WAMA     |   3   |
| Objets                                   | **pyzmq** PUB                     | RTMaps                 |   1+  |
| Vidéo world brute                        | (enregistrée par Pupil)           | —                      |   —   |
| Vidéo annotée (overlay)                  | pyzmq PUB                         | RTMaps                 |   5   |

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
- Export ZMQ PUB → RTMaps (objet observé + coordonnées de boîte). *(optionnel en v1)*

### Phase 2 — Segmentation (intégrée dès la v1)
- `yolo11n-seg`, overlay des masques, matching point-dans-polygone (`cv2.pointPolygonTest`).
- Gain qualitatif : « regarde la voiture » vs « regarde le ciel derrière la boîte ».

### Phase 3 — LSL
- Outlet `pylsl` des données objets (côté serveur), horloge `local_clock()`.
- Base de la synchro multi-sources pour l'intégration WAMA (gaze + véhicule + EEG/GSR…).

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
     `objects.pldata` (masques complets pour tous les objets → relecture fidèle). Traite a
     posteriori des données oculo déjà acquises sans le plugin. Idéal pour SAM3 hors temps réel.
- Le matching gaze réutilise la même logique point-dans-boîte / point-dans-masque que Capture.
- ⚠️ À vérifier en live : rendu Player (`frame.img` vs `gl_display`) et API
  `PLData_Writer`/`load_pldata_file` du bundle (codés selon l'API standard, non exécutés ici).

### Phase 4bis — Sélection des classes (fait, Capture)
- Filtre d'affichage par classe dans le plugin (`classes_filter`, liste blanche temps réel) : YOLO
  détecte tout (aucun impact perf), **toutes** les détections restent enregistrées, seul l'overlay
  et l'éligibilité au focus sont filtrés. Liste des classes vues affichée pour la découverte.

### Phase 5 — Vidéo annotée
- Stream/enregistrement de la vidéo surchargée de l'overlay (pyzmq → RTMaps), si pertinent.

---

## Découpage des fichiers

| Fichier                      | Env    | Rôle                                                       |
|------------------------------|:------:|------------------------------------------------------------|
| `yolo_server.py`             | 3.12   | Serveur ZMQ REP : frame → YOLO → détections/masques        |
| `detection_plugin.py`        | 3.6    | Plugin Pupil Capture mince (client ZMQ, gaze, overlay, IPC)|
| `requirements-detector.txt`  | 3.12   | Dépendances du venv détecteur                              |
| `rtmaps_stream.py`           | RTMaps | Réception ZMQ côté RTMaps (réécrit en pyzmq nu) — phase 1+ |
| `~Archives/`                 | —      | `darknet`, `imagezmq`, ancien plugin (nettoyage versionning ultérieur) |

---

## Installation (cible)

1. **Détecteur** (une fois, dans le venv 3.12) :
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements-detector.txt
   ```
2. **Plugins** :
   - Capture : copier `detection_plugin.py` dans `~/pupil_capture_settings/plugins/`.
   - Player : copier `player_object_recognition.py` dans `~/pupil_player_settings/plugins/`.
3. **Lancement** : démarrer `python yolo_server.py` puis Pupil Capture/Player ; activer le plugin
   « Object Recognition (YOLO) » dans le Plugin Manager.
   - Player : si le recording contient déjà `objects.pldata` → relecture auto de l'overlay ; sinon
     bouton **Reprocess recording** (détecteur lancé) pour le générer.
