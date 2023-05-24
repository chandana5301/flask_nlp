FROM windows:latest
RUN yum install net-tools -y
RUN yum install httpd -y
RUN yum install python3 -y
RUN pip3 install -r requirements.txt
WORKDIR flask_nlp
ENTRYPOINT ["python3", "apps.py"]
EXPOSE 3000 5050