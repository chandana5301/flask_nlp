FROM windows:latest
RUN yum install net-tools -y
RUN yum install httpd -y
RUN yum install python3 -y
COPY requirements.txt /Nlp_Deployment
RUN pip3 install -r /Nlp_Deployment/requirements.txt
COPY ../Nlp_Deployment Nlp_Deployment/nlpmodel
WORKDIR nlpmodel
ENTRYPOINT ["python3", "app.py"]
EXPOSE 3000 5050