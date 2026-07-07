# Integration tests

Run on inferno (GPU + ComfyUI + ~/comfy scripts required):

    rsync -a --exclude .venv --exclude cache --exclude .git ./ inferno:~/rr-ai-gateway/
    ssh inferno 'cd ~/rr-ai-gateway && python3 -m venv .venv && \
        .venv/bin/pip install -r requirements.txt -r requirements-dev.txt && \
        mkdir -p tests/integration/fixtures && \
        cp <some-subject-photo>.jpg tests/integration/fixtures/subject.jpg && \
        COMFY_PORT=8188 .venv/bin/pytest -m integration -v'

ComfyUI on inferno listens on 8188 (engine default is 5545), hence COMFY_PORT=8188.
