FROM python:3.8.15

COPY train_model.py .

# install packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "train_model.py"]