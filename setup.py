from setuptools import setup, find_packages

setup(
    name='knowledgegraph_chatbot',
    version='0.1.0',  # Consider using semantic versioning for clarity
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
            'knowledgegraph_chatbot=knowledgegraph_chatbot.chatbot:main',  # Ensure the 'main' function exists
        ],
    },
    author='Subha', 
    author_email='subhasumaiya@gmail.com',
    description='A package for integrating language models with a Neo4j graph database.',
    long_description=open('README.md').read(),  # Read from README.md for package description
    long_description_content_type='text/markdown',  # Format of the long description
    url='https://github.com/subhajamal/KnowledgeGraphChatbot',  
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12', 
    ],
    python_requires='>=3.7',  
)
