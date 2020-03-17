FROM python:3.8

COPY . /app

WORKDIR /app

RUN pip3 install pipenv
RUN pipenv install

ENV FLASK_APP=server.py

COPY . .

EXPOSE 8080

CMD ["pipenv", "run", "flask", "run"]