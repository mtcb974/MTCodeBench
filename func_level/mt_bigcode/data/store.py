import json

def store_data(mt_data_list,file_path):

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(mt_data_list,file,ensure_ascii=False,indent=4)
    
    print(f"All instances have been saved to {file_path}")