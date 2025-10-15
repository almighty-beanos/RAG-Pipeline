# RAG-Pipeline

## Setup
Ensure Python is running in your system. If not, install any version of Python after 3.x.x, Ascertain by running
```bash
python --version
```
Now, run the following in your folder for installation of libraries and frameworks for functioning of the pipeline. You should see the list of all the installed requirements of the pipeline
```bash
pip install -r requirements.txt
pip list
```
Upon requirement installation, run the following to execute the application
```bash
uvicorn app:app
```
## Evaluation
The system accepts a starting URL to crawl within N domain pages and stores the indexing and create vector embeddings of the document chunks which is storec along with the metadata of the vectorised chunks. It also responds to user query in order to obtain information regarding the website.
