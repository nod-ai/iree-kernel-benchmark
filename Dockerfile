FROM python:3.11

ENV DASH_DEBUG_MODE False
COPY ./frontend /frontend
WORKDIR /frontend
RUN set -ex && \
    pip install -r requirements.txt
EXPOSE 8050
CMD ["gunicorn", "--workers=10", "--threads=4", "-b", "0.0.0.0:8050", "--reload", "kernel_visualizer:server"]
