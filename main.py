





from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends,Body
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
import shutil
import json
import tempfile
from datetime import datetime
from country_mapping import country_mapping
from doc_intelligence_with_formatting import basic_openai,certificate_openai, experience_openai, reposition_fields, validate_parsed_resume, extract_resume_info, replace_values, replace_rank, convert_docx_to_pdf,replace_country
from rank_map_dict import rank_mapping
from dict_file import mapping_dict
import os
import asyncio
from current_doc_parsor import process_document_to_json



import pandas as pd
import io
import time
import schedule
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from sqlalchemy import text
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz




from dotenv import load_dotenv
import re


load_dotenv()
app = FastAPI(title="Resume Parser API", version="1.0")


# Secure API Key Authentication
API_KEY = os.getenv("your_secure_api_key")
API_KEY_NAME = os.getenv("api_key_name")
endpoint = os.getenv("endpoint")
key = os.getenv("key")
model_id = os.getenv("model_id")
container_name = os.getenv("container_name")
connection_string = os.getenv("connection_string")

basic_details_order = [
    "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", 
    "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", "City", 
    "State", "Country", "ZipCode", "EmailId", "MobileNo", "AlternateNo", "Rank"
]


experience_table_order = [
    "VesselName", "VesselType", "Position", "VesselSubType", "Employer", 
    "Flag", "IMO", "FromDt", "ToDt", "others"
]

certificate_table_order = [
    "CertificateNo", "CertificateName", "PlaceOfIssue", "IssuedBy", "DateOfIssue", 
    "DateOfExpiry", "Grade", "Others", "CountryOfIssue"
]

# Define API Key Security
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

    
def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API Key"""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail=" Invalid API Key")
    return api_key


def clean_vessel_names(experience_table):
    if not experience_table or len(experience_table) < 2:
        return experience_table

    # Identify the header index for "VesselName"
    header = experience_table[0]
    vessel_name_index = None
    for key, value in header.items():
        if value.lower() == "vesselname":
            vessel_name_index = key
            break

    if vessel_name_index is None:
        return experience_table  # Skip if "VesselName" not found

    # Updated pattern includes hyphens after prefix
    prefix_pattern = r'^(M[\s./\\]?[V|T][\s.\-]*)'

    # Clean each row
    for row in experience_table[1:]:
        vessel_name = row.get(vessel_name_index, "")
        if vessel_name:
            # Remove the prefix if it matches
            cleaned_name = re.sub(prefix_pattern, '', vessel_name, flags=re.IGNORECASE).strip()
            row[vessel_name_index] = cleaned_name

    return experience_table

@app.post("/upload/")
async def upload_file(
    api_key: str = Depends(verify_api_key),  # Enforce API key authentication
    file: UploadFile = File(...), 
    entity: str = Form("")
):
    try:
        # Extract file extension
        suffix = os.path.splitext(file.filename)[-1]
        if suffix not in [".pdf", ".docx"]:
            raise HTTPException(status_code=400, detail="Only PDF and Word documents are allowed")

        # Generate custom filename with timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{timestamp}{suffix}")

        # Write the uploaded file to the custom temp path
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Handle .docx conversion if needed
        if suffix == ".docx":
            temp_file_path = await convert_docx_to_pdf(temp_file_path)  

        # Extract JSON from document
        extracted_info = extract_resume_info(endpoint, key, model_id, temp_file_path)
        # print(extracted_info)
        # print("-------------------------------------------------extracted text-------------------------------------------------","\n")


        fields_only = extracted_info["fields"]
        # print("fields")
        # print(fields_only)


        tables = extracted_info.get('tables', [])
        certificate_table = None
        experience_table = None
        
        for table in tables:
            if table.get('table_name') == 'certificate_table':
                certificate_table = table
            elif table.get('table_name') == 'experience_table':
                experience_table = table

        

        print("Certificate Table:")
        print(certificate_table)

        print("\nExperience Table:")
        print(experience_table)

        basic_out, cert_out, expe_out = await asyncio.gather(
            asyncio.to_thread(basic_openai, fields_only),
            asyncio.to_thread(certificate_openai, certificate_table),
            asyncio.to_thread(experience_openai, experience_table),
        )

        # basic_out = basic_openai(fields_only)
        # cert_out = certificate_openai(certificate_table)
        # expe_out = experience_openai(experience_table)

        
        basic_details_merge = basic_out['basic_details']
        print("basic_details_merge")
        print(basic_details_merge)
        certificate_table_merge = cert_out['certificate_table']
        print("certificate_table_merge")
        print(certificate_table_merge)




         # --- Inject Gender Logic ---
        if basic_details_merge and len(basic_details_merge) > 1:
            header = basic_details_merge[0]
            values = basic_details_merge[1]

            gender_index = None
            for keyy, value in header.items():
                if value.lower() == "gender":
                    gender_index = keyy
                    break

            if gender_index is not None:
                gender_value = str(values.get(gender_index, "")).strip().lower()

                if gender_value not in ["male", "female"]:
                    values[gender_index] = "Male"
                else:
                    values[gender_index] = gender_value.capitalize()
        # --------------------------

        


        
        
        # experience_table_merge = expe_out['experience_table']
        # print("experience_table_merge")
        # print(experience_table_merge)
        
        
        
        
        experience_table_merge = expe_out['experience_table']
        experience_table_merge = clean_vessel_names(experience_table_merge)
        print("experience_table_merge")
        print(experience_table_merge)



        desired_basic_order = [
            "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", 
            "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", 
            "City", "State", "Country", "ZipCode", "EmailId", "MobileNo", 
            "AlternateNo", "Rank"
        ]

        desired_cert_order = [
            "CertificateNo",       # 0
            "CertificateName",     # 1
            "PlaceOfIssue",        # 2
            "IssuedBy",            # 3
            "DateOfIssue",         # 4
            "DateOfExpiry",        # 5
            "Grade",               # 6
            "Others",              # 7
            "CountryOfIssue"       # 8
        ]


        desired_experience_order = [
            "VesselName",     # 0
            "VesselType",     # 1
            "Position",       # 2
            "VesselSubType",  # 3
            "Employer",       # 4
            "Flag",           # 5
            "IMO",            # 6
            "FromDt",         # 7
            "ToDt",           # 8
            "others"          # 9
        ]

        def reorder_basic_details_table(data):
            if not data:
                return []

            # Get current header
            current_header = data[0]
            
            # Map field name to current index
            name_to_index = {v: k for k, v in current_header.items()}
            
            # New header in desired order
            new_header = {str(i): field_name for i, field_name in enumerate(desired_basic_order)}

            reordered_data = [new_header]

            for row in data[1:]:
                new_row = {}
                for new_idx, field_name in enumerate(desired_basic_order):
                    old_idx = name_to_index.get(field_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)

            return reordered_data


        def reorder_experience_table(data):
            if not data:
                return []

            # Get the current header (mapping of index to field names)
            current_header = data[0]
            
            # Create a mapping from field name to current index
            name_to_index = {v: k for k, v in current_header.items()}
            
            # Create new header in desired order
            new_header = {str(i): field_name for i, field_name in enumerate(desired_experience_order)}

            # Rebuild the table in the new order
            reordered_data = [new_header]

            for row in data[1:]:
                new_row = {}
                for new_idx, field_name in enumerate(desired_experience_order):
                    old_idx = name_to_index.get(field_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)

            return reordered_data
        

        def reorder_certificate_table(data):
            if not data:
                return []

            # Step 1: Get current header (first element is a dict with index keys)
            current_header = data[0]
            
            # Build mapping: "CertificateNo" => "0", etc.
            name_to_index = {v: k for k, v in current_header.items()}
            
            # Step 2: Build new header with desired order
            new_header = {str(i): col_name for i, col_name in enumerate(desired_cert_order)}
            
            # Step 3: Reorder all rows based on desired order
            reordered_data = [new_header]
            
            for row in data[1:]:
                new_row = {}
                for new_idx, col_name in enumerate(desired_cert_order):
                    old_idx = name_to_index.get(col_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)

            return reordered_data


        reordered_basic = reorder_basic_details_table(basic_details_merge)
        print(reordered_basic)
        reordered_certificates = reorder_certificate_table(certificate_table_merge)
        print(reordered_certificates)
        reordered_experience = reorder_experience_table(experience_table_merge)
        print(reordered_experience)

            
        final_output = {
            "status": "success",
            "data": {
                "basic_details": reordered_basic,
                "experience_table": reordered_experience,
                "certificate_table": reordered_certificates
            },
            "utc_time_stamp": datetime.utcnow().strftime("%d/%m/%Y, %H:%M:%S")
        }


        validation_errors = validate_parsed_resume(extracted_info, temp_file_path, 0.8, container_name, connection_string)
        print(validation_errors)



        course_map = replace_values(final_output, mapping_dict)
        rank_map = replace_rank(course_map, rank_mapping)
        rank_map=replace_country(rank_map,country_mapping)
        final_output['data']['basic_details'] = replace_country(rank_map['data']['basic_details'], country_mapping)
        final_output['data']['certificate_table'] = replace_country(rank_map['data']['certificate_table'], country_mapping)

        # rank_map_dict = json.loads(rank_map)


        
        basic_details = rank_map.get('data', {}).get('basic_details', [])
        experience_table = rank_map.get('data', {}).get('experience_table', [])
        certificate_table = rank_map.get('data', {}).get('certificate_table', [])

        # Reposition columns only if the section exists
        if basic_details:
            basic_details = reposition_fields(basic_details, basic_details_order)
        if experience_table:
            experience_table = reposition_fields(experience_table, experience_table_order)
        if certificate_table:
            certificate_table = reposition_fields(certificate_table, certificate_table_order)

        # Update the input_json with the new order (only if they exist)
        rank_map.setdefault('data', {})
        rank_map['data']['basic_details'] = basic_details
        rank_map['data']['experience_table'] = experience_table
        rank_map['data']['certificate_table'] = certificate_table 

        # # transformed_data = send_to_gpt(rank_map)

        experience_table = rank_map["data"]["experience_table"]
        if experience_table:
            # Keep the header row intact
            filtered_experience = [experience_table[0]]
    
            # Define the null threshold (remove if more than 8 are null out of 10)
            max_allowed_nulls = 8
    
            # Filter rows
            for row in experience_table[1:]:
                # Treat as valid row only if it's a full record with limited nulls
                if isinstance(row, dict):
                    null_count = sum(1 for v in row.values() if v is None)
                    if null_count < max_allowed_nulls:
                        filtered_experience.append(row)
    
            # Update the data with filtered experience_table
            rank_map["data"]["experience_table"] = filtered_experience
            print(filtered_experience)
        return rank_map
        
    except HTTPException as http_exc:
        # FastAPI will return this as-is
        raise http_exc

    except Exception as e:
        # Catch-all for unexpected errors
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during resume processing.",
                "detail": str(e)
            }
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)






# Endpoint for uploading the file
@app.post("/upload-document/")
async def upload_file(api_key: str = Depends(verify_api_key), file: UploadFile = File(...), Doctype: str = Form("")):
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".jpg", ".jpeg", ".png", ".docx"]:
        return JSONResponse({"error": "Only PDF, DOCX, or image files allowed."}, status_code=400)

    # Handle DOCX conversion if the file is a DOCX
    if file_ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        try:
            pdf_path = await convert_docx_to_pdf(temp_file_path)
            result = process_document_to_json(pdf_path)
            # return JSONResponse(content=result, media_type="application/json")
            # if isinstance(result, str):
            #     result = json.loads(result)

            # Normalize and flatten the mapping dictionary
            normalized_mapping = {}
            for key, value in mapping_dict.items():
                aliases = [alias.strip().lower() for alias in key.split("/")]
                for alias in aliases:
                    normalized_mapping[alias] = value

            # Perform docName mapping
            for item in result:
                original_doc_name = item.get("docName", "").strip().lower()
                mapped_doc_name = normalized_mapping.get(original_doc_name)
                if mapped_doc_name:
                    item["docName"] = mapped_doc_name

            return JSONResponse(content=result, media_type="application/json")
        
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            os.remove(temp_file_path)
    else:
        # Handle PDF and image files as before
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            result = process_document_to_json(temp_file_path)

            # If result is a JSON string, convert it to a Python object
            if isinstance(result, str):
                result = json.loads(result)

            # Normalize and flatten the mapping dictionary
            normalized_mapping = {}
            for key, value in mapping_dict.items():
                aliases = [alias.strip().lower() for alias in key.split("/")]
                for alias in aliases:
                    normalized_mapping[alias] = value

            # Perform docName mapping
            for item in result:
                original_doc_name = item.get("docName", "").strip().lower()
                mapped_doc_name = normalized_mapping.get(original_doc_name)
                if mapped_doc_name:
                    item["docName"] = mapped_doc_name

                original_country = item.get("issuedCountry", "").strip().lower()
                for key, value in country_mapping.items():
                    if key.strip().lower() == original_country:
                        item["issuedCountry"] = value
                        break

            return JSONResponse(content=result, media_type="application/json")

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            os.remove(temp_file_path)





# Set the path for your local SQLite database file
sqlite_db_path = "Resume_Parser.db"

# Define base for SQLAlchemy ORM mapping
Base = declarative_base()

# Database connection (SQLite)

def get_db_engine():
    # Change the connection string to use SQLite
    connection_string = f"sqlite:///{sqlite_db_path}"
    return create_engine(connection_string)

# Check if tables exist, and create them if they don't

def create_tables():
    engine = get_db_engine()
    inspector = inspect(engine)
    # Check if the tables already exist
    if not inspector.has_table('temp_table'):
        Base.metadata.create_all(engine)  # Create tables if they don't exist

# Define the tables using SQLAlchemy ORM (for SQLite)

class TempDocument(Base):
    __tablename__ = 'temp_table'

    id = Column(Integer, primary_key=True, autoincrement=True)
    unidentified_doc_name = Column(String, nullable=False)
    mapped_doc_name = Column(String, nullable=False)
    status = Column(String, default='pending')
    CreatedDate = Column(DateTime, default=datetime.now)

class MasterDocument(Base):
    __tablename__ = 'Master_unidentified_doc_Table'

    id = Column(Integer, primary_key=True, autoincrement=True)
    unidentified_doc_name = Column(String, nullable=False)
    mapped_doc_name = Column(String, nullable=False)
    uploaded_date = Column(DateTime, default=datetime.now)
    status = Column(String, default='pending')

# Initialize the tables at startup, but only if they don't exist
create_tables()

# API Input Model
class DocumentMapping(BaseModel):
    unidentified_doc_name: str
    mapped_doc_name: str



@app.post("/insert-temp-documents/")
def insert_temp_documents(
    api_key: str = Depends(verify_api_key),
    mappings: List[Dict[str, str]] = Body(...),
):
    try:
        if not mappings or not isinstance(mappings[0], dict):
            return {"error": "Invalid format. Expected a dictionary inside a list."}

        # Use the entire mappings list
        df = pd.DataFrame([
            {"unidentified_doc_name": item["unidentified_doc_name"], "mapped_doc_name": item["mapped_doc_name"]}
            for item in mappings
        ])
        
        df['status'] = 'pending'
        df['CreatedDate'] = datetime.now()

        engine = get_db_engine()
        df.to_sql('temp_table', con=engine, if_exists='append', index=False)

        return {"message": f"{len(df)} document(s) inserted as pending."}
    except Exception as e:
        return {"error": f"Insertion failed: {str(e)}"}




# Scheduled Task: Export to Excel and upload to Blob

def export_data_to_excel():
    try:
        engine = get_db_engine()
        with engine.begin() as conn:
            # Select pending docs from previous dates
            result = conn.execute(text("""
                SELECT * FROM temp_table 
                WHERE TRIM(status) = 'pending' 
            """))
            data = result.fetchall()
            print(f"Fetched Data from temp_table: {data}")
            if not data:
                return  # Nothing to export

            df = pd.DataFrame(data, columns=result.keys())

            # Export to Excel (in-memory)
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            # Upload to Azure Blob Storage with 'exported' in name
            blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
            blob_name = f"verification_documents_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            blob_client = blob_service_client.get_blob_client(container=os.getenv("AZURE_BLOB_CONTAINER_NAME"), blob=blob_name)
            blob_client.upload_blob(excel_buffer, overwrite=True)

            # Update and delete exported rows from temp_table
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE temp_table SET status = 'exported' 
                    WHERE status = 'pending' 
                """))
                conn.execute(text("DELETE FROM temp_table WHERE status = 'exported'"))

    except Exception as e:
        print(f"Export Error: {str(e)}")

# Get latest replied blob (manual reply must be uploaded to blob)

def get_latest_replied_blob_name():
    """
    Fetch the most recently modified blob that represents a replied document.
    Replied files must follow the naming: verification_documents_replied_YYYYMMDD.xlsx
    """
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    container_client = blob_service_client.get_container_client(os.getenv("AZURE_BLOB_CONTAINER_NAME"))

    # Filter only replied files
    blobs = list(container_client.list_blobs(name_starts_with="verification_documents_replied_"))
    replied_blobs = sorted(
        [b for b in blobs if b.name.endswith(".xlsx")],
        key=lambda b: b.last_modified,
        reverse=True
    )

    if not replied_blobs:
        raise Exception("No replied verification documents found in blob storage.")

    return replied_blobs[0].name

def insert_data_from_blob(blob_name: str):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        blob_data = blob_service_client.get_blob_client(container=os.getenv("AZURE_BLOB_CONTAINER_NAME"), blob=blob_name).download_blob().readall()
        df = pd.read_excel(io.BytesIO(blob_data))

        # Insert into master table (SQLite-compatible schema)
        engine = get_db_engine()
        with engine.begin() as conn:
            for _, row in df.iterrows():
                # Check if the record already exists in Master_unidentified_doc_Table
                existing = conn.execute(text("""
                    SELECT 1 FROM Master_unidentified_doc_Table 
                    WHERE unidentified_doc_name = :unidentified_doc_name 
                    AND mapped_doc_name = :mapped_doc_name 
                    AND uploaded_date = :uploaded_date
                """), {
                    "unidentified_doc_name": row['unidentified_doc_name'],
                    "mapped_doc_name": row['mapped_doc_name'],
                    "uploaded_date": row['CreatedDate']
                }).fetchone()

                if not existing:  # Insert only if the record does not exist
                    conn.execute(text("""
                        INSERT INTO Master_unidentified_doc_Table
                        (unidentified_doc_name, mapped_doc_name, uploaded_date, status)
                        VALUES (:unidentified_doc_name, :mapped_doc_name, :uploaded_date, :status)
                    """), {
                        "unidentified_doc_name": row['unidentified_doc_name'],
                        "mapped_doc_name": row['mapped_doc_name'],
                        "uploaded_date": row['CreatedDate'],
                        "status": row['status']
                    })

        print(f"Data from {blob_name} inserted successfully.")

    except Exception as e:
        print(f"Insertion from blob failed: {str(e)}")
# Combined scheduled task

def run_both_tasks():
    """
    1. Export pending records from temp_table to Azure blob as 'exported'
    2. Look for latest replied Excel from blob storage and insert into master table
    """
    export_data_to_excel()
    try:
        latest_blob = get_latest_replied_blob_name()
        insert_data_from_blob(latest_blob)
    except Exception as e:
        print(f"Insert skipped: {str(e)}")




@app.get("/view-temp-documents/")
def view_temp_documents(api_key: str = Depends(verify_api_key)):
    try:
        engine = get_db_engine()
        # Open a session to query the database
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        # Query all records from temp_table
        result = session.query(TempDocument).all()

        # Convert the result to a list of dictionaries, converting datetime to string
        data = [{
            "id": record.id,
            "unidentified_doc_name": record.unidentified_doc_name, 
            "mapped_doc_name": record.mapped_doc_name,
            "status": record.status, 
            "CreatedDate": record.CreatedDate.strftime('%Y-%m-%d %H:%M:%S') if record.CreatedDate else None
        } for record in result]

        # Return the data as a JSON response
        return JSONResponse(content={"data": data})

    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}



@app.post("/delete-temp-documents/")
def delete_temp_documents(api_key: str = Depends(verify_api_key)):
    try:
        engine = get_db_engine()

        # Use engine.begin() for automatic transaction handling and committing
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM temp_table"))
        
        return {"message": "All documents from temp_table have been deleted."}
    except Exception as e:
        return {"error": f"Deletion failed: {str(e)}"}






scheduler = AsyncIOScheduler(timezone=pytz.timezone("Asia/Kolkata"))

@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(
        run_both_tasks,
        CronTrigger(hour=19, minute=15),  # 19:15 PM IST daily
        id="run_both_tasks_daily",
        replace_existing=True # prevents immediate run
    )
    scheduler.start()
    print("APScheduler started: daily 19:15 Asia/Kolkata")
    job = scheduler.get_job("run_both_tasks_daily")
    print("Next run time (UTC):", job.next_run_time)

@app.on_event("shutdown")
async def shutdown_scheduler():
    scheduler.shutdown(wait=False)
    print("APScheduler stopped")






# Startup scheduler (background)

# @app.on_event("startup")
# async def start_scheduler():
#     def run_scheduler():
#         schedule.every().monday.at("00:00").do(run_both_tasks)
#         while True:
#             schedule.run_pending()
#             time.sleep(60)
#     import threading
#     threading.Thread(target=run_scheduler, daemon=True).start()
