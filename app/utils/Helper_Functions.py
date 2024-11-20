from uuid import uuid4
def create_text_from_dict(item):
    """
    Create a text string from a dictionary, joining each key-value pair
    into a readable sentence format.
    """
    text_parts = []

    # Iterate through each key-value pair
    for key, value in item.items():
        # Convert value to string if it isn't already (e.g., for numerical or list data)
        value_str = str(value)

        # Make the key more readable (e.g., replacing underscores with spaces and capitalizing)
        key_readable = key.replace('_', ' ').capitalize()

        # Add the key-value pair to the list in a readable format
        text_parts.append(f"{key_readable}: {value_str}")

    # Join all parts into a single string
    return ', '.join(text_parts)

def create_uuid4_from_list(lst):
    return [str(uuid4()) for _ in range(len(lst))]