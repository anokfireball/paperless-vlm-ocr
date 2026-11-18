# Data-only image: the OCI image carries the plugin site tree
# (package + dist-info) and is mounted as an image volume
# (`persistence: type: image` in app-template). It is never run as a container.
FROM scratch
COPY site/ /
