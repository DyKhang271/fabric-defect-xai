
# Fabric Defect Detection with YOLO and XAI

Detect nine types of surface defects in fabric images and inspect which parts of an image influenced a detection. The project includes a YOLO26 training notebook, a Flask interface, and scripts for explanation and evaluation.

- **Detection:** The Flask app combines detections from three checkpoints using weighted boxes fusion.
- **Explanation:** Occlusion sensitivity measures how masking image patches changes the confidence of a matched detection. Grad-CAM provides a second heatmap.
- **Evaluation:** The scripts compare detections with annotations using IoU and summarize how much heat falls inside the reference box.

A [saved example](results/example-xai-metrics.json) shows a prediction with IoU **0.655** whose explanations put only **12.3%** of occlusion heat and **16.4%** of Grad-CAM heat inside the matched annotation. This is one example of localization and explanation disagreeing. It is not a benchmark average.

## Run locally

From the repository root, install the dependencies in a Python virtual environment.

```shell
python -m pip install -r requirements.txt
```

For single-model inference, use a YOLO checkpoint trained for the label map in [dataset.yaml](voc2007_yolo/dataset.yaml).

```shell
python Code/xai/yolo_xai_llm.py --weights path/to/best.pt --image path/to/fabric.jpg --xai-method both
```

The Flask app uses the three checkpoints listed in [ensemble_config.json](Code/xai/conf/ensemble_config.json). Place your compatible checkpoints at those paths or update the config before launching the app.

```shell
python Code/xai/app_xai_llm.py --server-name 127.0.0.1 --server-port 7860
```

Open `http://127.0.0.1:7860` and upload an image. The optional LLM explanation can be left disabled.

## Training and evaluation

- [YOLO26 training notebook](notebooks/yolo26-fabric-training.ipynb) uses the Kaggle input `anhkhoa675/fabric-dataset-ver1`. Update the dataset paths for your environment.
- [Evaluation script](Code/xai/benchmark_xai.py) accepts paths to your checkpoint, dataset YAML, and test images. Run it with `--help` for the options.

Source code and one anonymized result are included. Supply the dataset images and checkpoints separately.
