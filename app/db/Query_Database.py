import mysql.connector
import os


def Query_Database(query):
    try:
        # Establishing the connection
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )

        if connection.is_connected():
            print("Connected to the database.")

            # Creating a cursor to execute the query
            cursor = connection.cursor(dictionary=True)  # Use dictionary=True to get results as dictionaries
            cursor.execute(query)

            # Fetching the results
            results = cursor.fetchall()

            # Closing the cursor
            cursor.close()

            return results

    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None

    finally:
        if connection.is_connected():
            connection.close()
            print("Connection closed.")