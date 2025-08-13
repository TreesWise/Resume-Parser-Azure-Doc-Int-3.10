


# from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends,Body
# from fastapi.responses import JSONResponse
# from fastapi.security.api_key import APIKeyHeader
# import os
# import shutil
# import json
# import tempfile
# from datetime import datetime
# # from cv_json_docpage import cv_json
# from country_mapping import country_mapping
# from doc_intelligence_with_formatting import basic_openai,certificate_openai, experience_openai, reposition_fields, validate_parsed_resume, extract_resume_info, replace_values, replace_rank, convert_docx_to_pdf,replace_country
# from rank_map_dict import rank_mapping
# from dict_file import mapping_dict
# import os
# import tempfile
# import shutil
# import json
# from datetime import datetime
# import asyncio

# from dotenv import load_dotenv
# import re


# load_dotenv()

# app = FastAPI(title="Resume Parser API", version="1.0")


# # Secure API Key Authentication
# API_KEY = os.getenv("your_secure_api_key")
# API_KEY_NAME = os.getenv("api_key_name")
# endpoint = os.getenv("endpoint")
# key = os.getenv("key")
# model_id = os.getenv("model_id")
# container_name = os.getenv("container_name")
# connection_string = os.getenv("connection_string")

# basic_details_order = [
#     "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", 
#     "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", "City", 
#     "State", "Country", "ZipCode", "EmailId", "MobileNo", "AlternateNo", "Rank"
# ]


# experience_table_order = [
#     "VesselName", "VesselType", "Position", "VesselSubType", "Employer", 
#     "Flag", "IMO", "FromDt", "ToDt", "others"
# ]

# certificate_table_order = [
#     "CertificateNo", "CertificateName", "PlaceOfIssue", "IssuedBy", "DateOfIssue", 
#     "DateOfExpiry", "Grade", "Others", "CountryOfIssue"
# ]

# # Define API Key Security
# api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

    
# def verify_api_key(api_key: str = Security(api_key_header)):
#     """Validate API Key"""
#     if not api_key or api_key != API_KEY:
#         raise HTTPException(status_code=403, detail=" Invalid API Key")
#     return api_key


# def clean_vessel_names(experience_table):
#     if not experience_table or len(experience_table) < 2:
#         return experience_table

#     # Identify the header index for "VesselName"
#     header = experience_table[0]
#     vessel_name_index = None
#     for key, value in header.items():
#         if value.lower() == "vesselname":
#             vessel_name_index = key
#             break

#     if vessel_name_index is None:
#         return experience_table  # Skip if "VesselName" not found

#     # Updated pattern includes hyphens after prefix
#     prefix_pattern = r'^(M[\s./\\]?[V|T][\s.\-]*)'

#     # Clean each row
#     for row in experience_table[1:]:
#         vessel_name = row.get(vessel_name_index, "")
#         if vessel_name:
#             # Remove the prefix if it matches
#             cleaned_name = re.sub(prefix_pattern, '', vessel_name, flags=re.IGNORECASE).strip()
#             row[vessel_name_index] = cleaned_name

#     return experience_table

# @app.post("/upload/")
# async def upload_file(
#     api_key: str = Depends(verify_api_key),  # Enforce API key authentication
#     file: UploadFile = File(...), 
#     entity: str = Form("")
# ):
#     try:
#         # Extract file extension
#         suffix = os.path.splitext(file.filename)[-1]
#         if suffix not in [".pdf", ".docx"]:
#             raise HTTPException(status_code=400, detail="Only PDF and Word documents are allowed")

#         # Generate custom filename with timestamp
#         timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
#         temp_file_path = os.path.join(tempfile.gettempdir(), f"{timestamp}{suffix}")

#         # Write the uploaded file to the custom temp path
#         with open(temp_file_path, "wb") as temp_file:
#             shutil.copyfileobj(file.file, temp_file)

#         # Handle .docx conversion if needed
#         if suffix == ".docx":
#             temp_file_path = await convert_docx_to_pdf(temp_file_path)  

#         # Extract JSON from document
#         extracted_info = extract_resume_info(endpoint, key, model_id, temp_file_path)
#         # print(extracted_info)
#         # print("-------------------------------------------------extracted text-------------------------------------------------","\n")


#         fields_only = extracted_info["fields"]
#         # print("fields")
#         # print(fields_only)


#         tables = extracted_info.get('tables', [])
#         certificate_table = None
#         experience_table = None
        
#         for table in tables:
#             if table.get('table_name') == 'certificate_table':
#                 certificate_table = table
#             elif table.get('table_name') == 'experience_table':
#                 experience_table = table

        

#         print("Certificate Table:")
#         print(certificate_table)

#         print("\nExperience Table:")
#         print(experience_table)

#         basic_out, cert_out, expe_out = await asyncio.gather(
#             asyncio.to_thread(basic_openai, fields_only),
#             asyncio.to_thread(certificate_openai, certificate_table),
#             asyncio.to_thread(experience_openai, experience_table),
#         )

#         # basic_out = basic_openai(fields_only)
#         # cert_out = certificate_openai(certificate_table)
#         # expe_out = experience_openai(experience_table)

        
#         basic_details_merge = basic_out['basic_details']
#         print("basic_details_merge")
#         print(basic_details_merge)
#         certificate_table_merge = cert_out['certificate_table']
#         print("certificate_table_merge")
#         print(certificate_table_merge)
        
        
#         # experience_table_merge = expe_out['experience_table']
#         # print("experience_table_merge")
#         # print(experience_table_merge)
        
        
        
        
#         experience_table_merge = expe_out['experience_table']
#         experience_table_merge = clean_vessel_names(experience_table_merge)
#         print("experience_table_merge")
#         print(experience_table_merge)



#         desired_basic_order = [
#             "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", 
#             "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", 
#             "City", "State", "Country", "ZipCode", "EmailId", "MobileNo", 
#             "AlternateNo", "Rank"
#         ]

#         desired_cert_order = [
#             "CertificateNo",       # 0
#             "CertificateName",     # 1
#             "PlaceOfIssue",        # 2
#             "IssuedBy",            # 3
#             "DateOfIssue",         # 4
#             "DateOfExpiry",        # 5
#             "Grade",               # 6
#             "Others",              # 7
#             "CountryOfIssue"       # 8
#         ]


#         desired_experience_order = [
#             "VesselName",     # 0
#             "VesselType",     # 1
#             "Position",       # 2
#             "VesselSubType",  # 3
#             "Employer",       # 4
#             "Flag",           # 5
#             "IMO",            # 6
#             "FromDt",         # 7
#             "ToDt",           # 8
#             "others"          # 9
#         ]

#         def reorder_basic_details_table(data):
#             if not data:
#                 return []

#             # Get current header
#             current_header = data[0]
            
#             # Map field name to current index
#             name_to_index = {v: k for k, v in current_header.items()}
            
#             # New header in desired order
#             new_header = {str(i): field_name for i, field_name in enumerate(desired_basic_order)}

#             reordered_data = [new_header]

#             for row in data[1:]:
#                 new_row = {}
#                 for new_idx, field_name in enumerate(desired_basic_order):
#                     old_idx = name_to_index.get(field_name)
#                     new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
#                 reordered_data.append(new_row)

#             return reordered_data


#         def reorder_experience_table(data):
#             if not data:
#                 return []

#             # Get the current header (mapping of index to field names)
#             current_header = data[0]
            
#             # Create a mapping from field name to current index
#             name_to_index = {v: k for k, v in current_header.items()}
            
#             # Create new header in desired order
#             new_header = {str(i): field_name for i, field_name in enumerate(desired_experience_order)}

#             # Rebuild the table in the new order
#             reordered_data = [new_header]

#             for row in data[1:]:
#                 new_row = {}
#                 for new_idx, field_name in enumerate(desired_experience_order):
#                     old_idx = name_to_index.get(field_name)
#                     new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
#                 reordered_data.append(new_row)

#             return reordered_data
        

#         def reorder_certificate_table(data):
#             if not data:
#                 return []

#             # Step 1: Get current header (first element is a dict with index keys)
#             current_header = data[0]
            
#             # Build mapping: "CertificateNo" => "0", etc.
#             name_to_index = {v: k for k, v in current_header.items()}
            
#             # Step 2: Build new header with desired order
#             new_header = {str(i): col_name for i, col_name in enumerate(desired_cert_order)}
            
#             # Step 3: Reorder all rows based on desired order
#             reordered_data = [new_header]
            
#             for row in data[1:]:
#                 new_row = {}
#                 for new_idx, col_name in enumerate(desired_cert_order):
#                     old_idx = name_to_index.get(col_name)
#                     new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
#                 reordered_data.append(new_row)

#             return reordered_data


#         reordered_basic = reorder_basic_details_table(basic_details_merge)
#         print(reordered_basic)
#         reordered_certificates = reorder_certificate_table(certificate_table_merge)
#         print(reordered_certificates)
#         reordered_experience = reorder_experience_table(experience_table_merge)
#         print(reordered_experience)

            
#         final_output = {
#             "status": "success",
#             "data": {
#                 "basic_details": reordered_basic,
#                 "experience_table": reordered_experience,
#                 "certificate_table": reordered_certificates
#             },
#             "utc_time_stamp": datetime.utcnow().strftime("%d/%m/%Y, %H:%M:%S")
#         }


#         validation_errors = validate_parsed_resume(extracted_info, temp_file_path, 0.8, container_name, connection_string)
#         print(validation_errors)



#         course_map = replace_values(final_output, mapping_dict)
#         rank_map = replace_rank(course_map, rank_mapping)
#         rank_map=replace_country(rank_map,country_mapping)
#         final_output['data']['basic_details'] = replace_country(rank_map['data']['basic_details'], country_mapping)
#         final_output['data']['certificate_table'] = replace_country(rank_map['data']['certificate_table'], country_mapping)

#         # rank_map_dict = json.loads(rank_map)


        
#         basic_details = rank_map.get('data', {}).get('basic_details', [])
#         experience_table = rank_map.get('data', {}).get('experience_table', [])
#         certificate_table = rank_map.get('data', {}).get('certificate_table', [])

#         # Reposition columns only if the section exists
#         if basic_details:
#             basic_details = reposition_fields(basic_details, basic_details_order)
#         if experience_table:
#             experience_table = reposition_fields(experience_table, experience_table_order)
#         if certificate_table:
#             certificate_table = reposition_fields(certificate_table, certificate_table_order)

#         # Update the input_json with the new order (only if they exist)
#         rank_map.setdefault('data', {})
#         rank_map['data']['basic_details'] = basic_details
#         rank_map['data']['experience_table'] = experience_table
#         rank_map['data']['certificate_table'] = certificate_table 

#         # # transformed_data = send_to_gpt(rank_map)

#         experience_table = rank_map["data"]["experience_table"]
#         if experience_table:
#             # Keep the header row intact
#             filtered_experience = [experience_table[0]]
    
#             # Define the null threshold (remove if more than 8 are null out of 10)
#             max_allowed_nulls = 8
    
#             # Filter rows
#             for row in experience_table[1:]:
#                 # Treat as valid row only if it's a full record with limited nulls
#                 if isinstance(row, dict):
#                     null_count = sum(1 for v in row.values() if v is None)
#                     if null_count < max_allowed_nulls:
#                         filtered_experience.append(row)
    
#             # Update the data with filtered experience_table
#             rank_map["data"]["experience_table"] = filtered_experience
#             print(filtered_experience)
#         return rank_map
        
#     except HTTPException as http_exc:
#         # FastAPI will return this as-is
#         raise http_exc

#     except Exception as e:
#         # Catch-all for unexpected errors
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "message": "An unexpected error occurred during resume processing.",
#                 "detail": str(e)
#             }
#         )

#     finally:
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)



from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends
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



