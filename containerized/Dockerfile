FROM tensorflow/tensorflow:latest-py3
COPY . /server
RUN apt-get update
RUN apt-get --yes install libxext6
RUN apt-get --yes install libsm6 libxrender1 libfontconfig1 libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install -r /server/requirements.txt
EXPOSE 8080
WORKDIR /server
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "8080"]