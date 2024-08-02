from setuptools import setup, find_packages

setup(
    name='knowledgegraph_chatbot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'langchain_openai',
        'langchain',
        'streamlit',
        'neo4j',
        'python-dotenv',
    ],
    entry_points={
        'console_scripts': [
            'knowledgegraph_chatbot=knowledgegraph_chatbot.chatbot:main',
        ],
    },
)
