import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.getenv('NOTION_TOKEN'))
database_id = os.getenv('NOTION_DATABASE_ID')

print("🔍 PROPRIEDADES DO DATABASE NOTION:")
print("=" * 50)

try:
    response = notion.databases.retrieve(database_id=database_id)
    properties = response['properties']

    print(f"📊 Total de propriedades: {len(properties)}")
    print()

    for name, prop in properties.items():
        prop_type = prop['type']
        print(f"  '{name}': {prop_type}")

        # Mostrar opções para campos select
        if prop_type == 'select' and 'select' in prop:
            options = prop['select'].get('options', [])
            if options:
                print(f"    Opções: {[opt['name'] for opt in options]}")
        print()

except Exception as e:
    print(f"❌ Erro: {e}")
