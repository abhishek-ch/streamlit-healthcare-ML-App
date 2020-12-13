FROM jupyter/pyspark-notebook:latest
LABEL maintainer="Abhishek Choudhary @Abc"

EXPOSE 8501

COPY src ./src
COPY requirements.txt ./
COPY resource ./resource

RUN pip3 install -r requirements.txt

ENV PYSPARK_PYTHON=python3
ENTRYPOINT ["streamlit", "run"]
CMD ["src/app.py"]

