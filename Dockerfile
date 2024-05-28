FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN chmod 1777 /tmp

WORKDIR /app

# Copy only the requirements file, to cache the installation of dependencies
COPY requirements.txt /app/requirements.txt

# COPY DESCRIPTIONS
# install dependencies
RUN pip install -r requirements.txt

# Download everything from NLTK
RUN python -m nltk.downloader punkt

# Expose the port the app runs on
EXPOSE 8000

# Set a default port
ENV PORT=8000

# Copy the rest of your application
COPY . /app

ENV GROBID_URL='https://kermitt2-grobid.hf.space'
ENV SCICO_API_URL=None

# Change the working directory
WORKDIR /app/src

# Run a shell
CMD ["bash"]

# # Run the uvicorn server in the conda environment
# CMD ["conda", "run", "--no-capture-output", "-n", "docker_env", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]