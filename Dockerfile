# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Copy model ke dalam container
COPY ./output/serving_model/income-prediction-model /models/income-prediction-model
# Copy config prometheus ke dalam container
COPY ./config /config

ENV MODEL_NAME=income-prediction-model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/config/prometheus.config"
ENV PORT=8501

# Custom entrypoint agar aktif Prometheus monitoring
RUN echo '#!/bin/bash \n\n\
    env \n\
    tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --monitoring_config_file=${MONITORING_CONFIG} \
    "$@"' > /usr/bin/tf_serving_entrypoint.sh \
    && chmod +x /usr/bin/tf_serving_entrypoint.sh

# Expose port untuk REST API
EXPOSE ${PORT}

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]