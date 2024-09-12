## Vehicle Object Detection Using YOLOv8 with Streamlit and Docker

This project demonstrates vehicle object detection using YOLOv8 on the Vehicles-coco dataset from Roboflow. The model is deployed with Streamlit and containerized using Docker. The YOLO model has been converted to an ONNX format for faster and better inference performance.

## Dataset Used
The Vehicles-coco dataset includes 18,000 images annotated with four classes: car, bus, motorcycle, and truck. You can find the dataset on [Roboflow](https://universe.roboflow.com/vehicle-mscoco/vehicles-coco).

## Clone the Repository
```bash
git clone https://github.com/NourhanNabil/detect-and-classify-vehicles-in-urban-street-scenes.git
cd detect-and-classify-vehicles-in-urban-street-scenes
```

### Prerequisites
- Docker installed on your machine.
- Docker Compose installed on your machine.

**Docker**
Build and Run the Docker Containers:
```bash
docker-compose up -d
```

**Interactive Shell**
For interactive shell access while the container is running, you can use:
```bash
docker-compose exec app bash
```

**Access the App**
Open your web browser and go to `http://localhost:8501` to access the App.

**Shut Down the Containers**
```bash
docker-compose down # Stops and removes containers, networks, volumes, and other services.
docker-compose stop # Stops containers without removing them, allowing you to start them again later.
```

### Installation (Optional)
If you prefer to run the application locally without Docker, you can install the required Python dependencies with:
```bash
pip install -r requirements.txt
