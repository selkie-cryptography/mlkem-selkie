FROM nginx:alpine

RUN mkdir -p /app

COPY index.html /app/index.html
COPY apple-touch-icon.png favicon-16x16.png favicon-16x16-dark.png \
     favicon-32x32.png favicon-32x32-dark.png \
     selkie-solid-white-on-transparent.svg \
     selkie-solid-black-on-transparent.svg /app/
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
CMD ["/entrypoint.sh"]
