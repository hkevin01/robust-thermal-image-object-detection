# Robust Thermal-Image Object Detection System

## Project Overview
This project is a submission for the **6th Real World Surveillance Workshop at WACV 2026** - Robust Thermal-Image Object Detection Challenge. The goal is to build object detectors that maintain consistent performance across seasons, weather patterns, and day-night cycles using the LTDv2 dataset.

## Core Features
- **Multi-Object Detection**: Detect 4 object classes (Person, Bicycle, Motorcycle, Vehicle) in thermal imagery
- **Robust Performance**: Maintain accuracy despite thermal drift caused by ambient changes, sensor calibration, and weather dynamics
- **Temporal Consistency**: Ensure stable detection performance across 8+ months of data
- **Context-Aware Processing**: Leverage weather metadata (temperature, humidity, solar radiation) for improved detection
- **Domain Adaptation**: Handle drastic variations in object and background appearance over time

## Dataset: LTDv2 (Large-Scale Long-term Thermal Drift Dataset v2)
- **Scale**: 1+ million frames over 8 months
- **Annotations**: 6.8+ million bounding boxes
- **Classes**: Person, Bicycle, Motorcycle, Vehicle
- **Metadata**: Weather conditions, temperature, humidity, solar radiation
- **Variations**: Multiple seasons, weather patterns, day-night cycles
- **Source**: https://huggingface.co/datasets/vapaau/LTDv2

## Evaluation Metrics
The challenge ranking is based on:
- **Global mAP@0.5**: Overall detection accuracy
- **Coefficient of Variation**: Consistency of per-month mAP@0.5 scores
- **Final Score**: Product of global mAP@0.5 × CoV (balances performance and consistency)

## Technical Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch / TensorFlow
- **Object Detection**: YOLOv8/YOLOv9, Faster R-CNN, DETR, or custom architectures
- **Data Processing**: OpenCV, NumPy, Pandas, Albumentations
- **Metadata Handling**: Pandas, Scikit-learn
- **Experiment Tracking**: Weights & Biases (wandb) / MLflow
- **Containerization**: Docker
- **Version Control**: Git

## Project Goals
1. **Primary**: Achieve top-tier global mAP@0.5 with minimal performance variation across months
2. **Secondary**: Develop novel techniques for handling thermal drift in object detection
3. **Tertiary**: Create reusable components for thermal imaging in surveillance contexts

## Target Users
- Computer vision researchers
- Surveillance system developers
- Thermal imaging practitioners
- Challenge participants

## Important Dates
- **October 17, 2025**: Competition start, Development Phase begins
- **December 1, 2025**: Testing Phase starts, Development Phase ends
- **December 7, 2025**: Competition ends
- **December 14, 2025**: Paper submission deadline
- **January 9, 2026**: Camera-ready deadline

## Citation
```
@article{LTDv2_dataset,
    title={LTDv2: A Large-Scale Long-term Thermal Drift Dataset for Robust Multi-Object Detection in Surveillance},
    DOI={10.36227/techrxiv.175339329.95323969/v1},
    publisher={Institute of Electrical and Electronics Engineers (IEEE)},
    author={Parola, Marco and Aakerberg, Andreas and Johansen, Anders S and Nikolov, Ivan A and Cimino, Mario GCA and Nasrollahi, Kamal and Moeslund, Thomas B},
    year={2025},
}
```

## License
This project uses the LTDv2 dataset under CC-BY-NC-4.0 license.
