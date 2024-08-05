# app/Dockerfile

FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]