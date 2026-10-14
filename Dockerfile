FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-lock.txt /app/requirements-lock.txt
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY --chown=user:user . /app
RUN mkdir -p /app/data /app/artifacts && chown -R user:user /app
USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
# Render asigna el puerto en $PORT; 7860 queda como valor local y para Hugging Face.
CMD python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860}
