# PPE Detection

## Docker

In order to build and run the docker image, you can follow these steps:

Frist, inside this repository biuld the docker image:

### `docker build -t ppe-detection:latest .`

Use the following command to run the built docker image:

### `docker run --gpus all -it -v $(pwd):/app ppe-detection:latest `

*Note that you have to be in the main directory of this project in order to create volume with the docker container.

## Detection

In the docker container, you can run the folowing command to start using the ppe detection program:

### `python3 main.py --input_path '/app/test-vdo.mp4' --output_path 'app/output' --deivce '0'` 

The input can be both image or vdo file, you can use the provided testing image and vdo to test.

