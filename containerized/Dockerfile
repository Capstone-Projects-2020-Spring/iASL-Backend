FROM tensorflow/tensorflow:latest-py3
COPY . /server
RUN pip install -r /server/requirements.txt
EXPOSE 8080
WORKDIR /server
CMD flask run -h 0.0.0.0 -p 8080