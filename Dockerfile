# Inference image in the shape TIRA expects: read $inputDataset/pairs.jsonl, write $outputDir/answers.jsonl.
#   docker build -t av-sbert .
#   docker run --rm -v $PWD/data/test:/in -v $PWD/out:/out -v $PWD/runs/all-MiniLM-L12-v2/best:/model av-sbert \
#       av-predict -i /in -o /out --model /model
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[model]" \
 && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

ENTRYPOINT []
CMD ["av-predict", "--help"]
