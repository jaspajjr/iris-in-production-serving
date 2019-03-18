FROM python:3.6

COPY container-entrypoint.sh /entry
RUN chmod 0755 /entry

COPY requirements.txt /data/requirements.txt
RUN pip install -r /data/requirements.txt

CMD ["start"]
ENTRYPOINT ["/entry"]

ARG GIT_COMMIT=unspecified
LABEL GIT_COMMIT=$GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

COPY model-training /data