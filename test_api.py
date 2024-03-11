
# Runpod API
# POD_ID = 'v9acc3qxvlznmo'
# url =f'https://{POD_ID}-8000.proxy.runpod.net'

# url = "https://9b30-34-126-169-86.ngrok-free.app"

# Local test
url = "http://0.0.0.0:8000"

print("URL: ",url)


##############
question = "What are the differences in work expectations between generations?"
##############


context = """
'Generational differences in work expectations and values are a significant topic of discussion in organizations today. Specifically, the differences between baby boomers and millennials have garnered attention due to their coexistence in the workplace. These generational differences can impact management dynamics at various levels, including operational, team, and strategic levels. Understanding these differences and finding ways to navigate them is crucial for creating a harmonious and productive work environment.

1. Contrasting Work Values, Attitudes, and Preferences:
Baby Boomers:
- Work Values: Baby boomers tend to prioritize loyalty, dedication, and hard work. They often value stability and long-term commitment to their organizations.
- Attitudes: Baby boomers may have a more traditional approach to work, valuing hierarchy and authority. They may prefer a structured work environment and clear guidelines.
- Preferences: Baby boomers may prefer face-to-face communication and value personal relationships in the workplace. They may also prioritize financial stability and job security.

Millennials:
- Work Values: Millennials often prioritize work-life balance, personal growth, and purpose-driven work. They seek opportunities for learning and development.
- Attitudes: Millennials may have a more collaborative and inclusive approach to work. They value flexibility, autonomy, and a sense of purpose in their roles.
- Preferences: Millennials are comfortable with technology and may prefer digital communication channels. They value feedback and recognition and seek a positive and engaging work environment.

2. Factors Contributing to Variations:
a) Socioeconomic Factors: Socioeconomic factors such as technological advancements, economic conditions, and societal changes can shape the expectations and values of different generations. For example, baby boomers experienced significant economic growth and stability, while millennials entered the workforce during times of economic uncertainty.
b) Cultural Influences: Cultural factors, including societal norms, values, and beliefs, can also contribute to generational differences. Each generation grows up in a unique cultural context that influences their perspectives on work.
c) Life Experiences: Different life experiences, such as major historical events or technological advancements, can shape the values and attitudes of each generation. For example, baby boomers may have experienced significant social and political changes, while millennials grew up in the digital age.


"""

import requests

input_text = {
    "context": context,
    "question": question
}

# Function to fetch streaming response
def fetch_data():
    response = requests.post(f'{url}/inference', json=input_text, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        # Process the data (in this example, simply print the chunks)
        print(chunk.decode('utf-8'), end = '')

# Call the function to start fetching data
fetch_data()
