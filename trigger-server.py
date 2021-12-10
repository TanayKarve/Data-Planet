from flask import Flask, request
import requests, shlex
import subprocess
import time
import os
import shutil

app = Flask(__name__)

@app.route("/update")
def hello_world():


    '''
    curl -O -J -H "X-Dataverse-key:65e93d83-e1ca-448f-9526-d6ce49831f6e" http://dataverse-dev.localhost:8085/api/access/dataset/:persistentId?persistentId=doi:10.5072/FK2/JR0NOV
    '''
    
    API_TOKEN='65e93d83-e1ca-448f-9526-d6ce49831f6e' # update with actual value or read from a file
    SERVER = "http://dataverse-dev.localhost:8085/api/" # URl of the server
    persistentID = "10.5072/FK2/FIWM45" # DOI of the dataset
    persistentID = "10.5072/"+request.args.get('doi')
    print("---------------------------------------")
    print(persistentID)
    print("---------------------------------------")
    # persistentID = "10.5072/FK2/JR0NOV"
    # Step-0: Download the dataset using REST APIs
    
    
    downloadCmd = "curl -O -J -H \"X-Dataverse-key:" +API_TOKEN+ "\" " + SERVER + "access/dataset/:persistentId?persistentId"+"=doi:"+ persistentID # Change to value from get 
    
    downloadedZipFile="dataverse_files.zip"

    args = shlex.split(downloadCmd)
    process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print("Output of downloading the file:")
    print(stdout)
    print(stderr)

    # Step-0.1: Extract the zip file.

    filePath = "storage/"
    from zipfile import ZipFile
    with ZipFile(downloadedZipFile, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        listOfFileNames = list(filter(lambda x: x.endswith("MLModel.json"), zipObj.namelist()))
        filePath += listOfFileNames[0]
        if(len(listOfFileNames)==1):
            zipObj.extract(listOfFileNames[0], "storage")
    
    # Step-1 : Reading the metadata file from the config file (MLModel)
    # extract the name of the correct folder by splitting on "." character
    
    
    metaData = open(filePath, "r")
    updatedMetadata = metaData.read().replace("\n", " ")

    # Step-2 : generating the relevant metadata file
    
    metadata = r'''{"fields": [{"typeName":"dsDescriptionValue","multiple":false,"typeClass":"primitive","value":"'''+updatedMetadata +'''"}]}'''
    generatedFile = open("generate-metadata.json", "w")
    generatedFile.write(metadata)
    generatedFile.close()


    print(metadata)
    # Step-3 : updating the metadata with the above generated file 

    
    
    uploadMetadataCMD = "curl -H \"X-Dataverse-key:" +API_TOKEN+"\" -X PUT \""+ SERVER+"datasets/:persistentId/editMetadata/?persistentId=doi:"+persistentID + "&replace=true\" --upload-file generate-metadata.json"
    
    
    
    print("START")
    print(uploadMetadataCMD)
    
    # os.system(uploadMetadataCMD)
    
    #args = shlex.split(uploadMetadataCMD)
    #process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("RUN COMMAND")
    #stdout, stderr = process.communicate()
    
    print("COMMUNICATED")

    print("Output of writing the latest metadata:")
    #print(stdout)
    #print(stderr)
    
    
    url = "http://dataverse-dev.localhost:8085/api/datasets/:persistentId/editMetadata/?persistentId=doi:"+persistentID+"&replace=true"

    payload=metadata
    headers = {'X-Dataverse-key': '65e93d83-e1ca-448f-9526-d6ce49831f6e'}
    response = requests.request("PUT", url, headers=headers, data=payload)
    print(response.text)
    
    
    
    return "OKABC"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000)
