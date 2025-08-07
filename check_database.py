#!/usr/bin/env python3
"""
Check PostgreSQL database structure and knowledge graph storage
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def check_database():
    """Check database tables and knowledge graph data"""
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment")
        return
    
    print(f"🔌 Connecting to PostgreSQL...")
    print(f"   URL: {database_url.split('@')[1] if '@' in database_url else 'hidden'}")
    
    try:
        # Create engine and session
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check connection
        result = session.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"✅ Connected to PostgreSQL")
        print(f"   Version: {version.split(',')[0]}")
        
        # List all tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\n📊 Tables in database: {len(tables)}")
        for table in tables:
            print(f"   - {table}")
        
        # Check if db_users table exists
        if 'db_users' in tables:
            print(f"\n✅ Table 'db_users' exists")
            
            # Get columns
            columns = inspector.get_columns('db_users')
            print(f"\n📋 Columns in db_users:")
            for col in columns:
                print(f"   - {col['name']} ({col['type']})")
            
            # Check if knowledge_graph column exists
            kg_column = next((col for col in columns if col['name'] == 'knowledge_graph'), None)
            if kg_column:
                print(f"\n✅ Column 'knowledge_graph' exists (type: {kg_column['type']})")
            else:
                print(f"\n❌ Column 'knowledge_graph' NOT FOUND")
                print("   Run the /database/add-knowledge-graph-column endpoint to add it")
            
            # Count users
            user_count = session.execute(text("SELECT COUNT(*) FROM db_users")).scalar()
            print(f"\n👥 Total users: {user_count}")
            
            # Check users with knowledge graphs
            if kg_column:
                kg_count = session.execute(text("""
                    SELECT COUNT(*) 
                    FROM db_users 
                    WHERE knowledge_graph IS NOT NULL 
                    AND knowledge_graph::text != '{"entities": {}, "relations": []}'
                """)).scalar()
                print(f"🧠 Users with knowledge graphs: {kg_count}")
                
                # Show sample knowledge graph data
                if kg_count > 0:
                    print("\n📝 Sample knowledge graph data:")
                    sample = session.execute(text("""
                        SELECT id, email, knowledge_graph 
                        FROM db_users 
                        WHERE knowledge_graph IS NOT NULL 
                        AND knowledge_graph::text != '{"entities": {}, "relations": []}'
                        LIMIT 1
                    """)).fetchone()
                    
                    if sample:
                        print(f"   User ID: {sample[0]}")
                        print(f"   Email: {sample[1]}")
                        print(f"   Knowledge Graph:")
                        kg_data = sample[2]
                        print(json.dumps(kg_data, indent=4))
        else:
            print(f"\n❌ Table 'db_users' does not exist")
            print("   You need to run the database initialization first")
        
        # Check conversations table
        if 'db_conversations' in tables:
            conv_count = session.execute(text("SELECT COUNT(*) FROM db_conversations")).scalar()
            print(f"\n💬 Total conversations: {conv_count}")
        
        session.close()
        
    except Exception as e:
        print(f"\n❌ Database error: {e}")
        print("\nPossible issues:")
        print("1. Database URL is incorrect")
        print("2. Database is not accessible")
        print("3. Tables haven't been created yet")

if __name__ == "__main__":
    check_database()