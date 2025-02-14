# Docker Usage

An official Docker image is not yet published (it is published internally at JPL), but may be built using the provided Dockerfile.

Build:
```bash
docker build \
  --no-cache \
  -t vic2png:develop \
  -f Dockerfile .
```

The intended use case is to expose local files to the Docker container using the `-v` flag and run the program via its entrypoint in a "single-shot" manner:
```bash
docker run \
  -v path/to/input_data:/stage-in \
  -v path/to/output_data:/stage-out \
  vic2png:develop \
  /stage-in/example.VIC \
  -o /stage-out/example.jpg
```

Be sure to replace the placeholder mounting directories with values appropriate for your use case.