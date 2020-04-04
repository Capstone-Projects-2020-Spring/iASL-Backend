FROM tensorflow/tensorflow:latest-py3
COPY . /server
RUN pip install -r /server/requirements.txt
EXPOSE 8080
WORKDIR /server
RUN ls -l
ENTRYPOINT ["flask", "run"]
