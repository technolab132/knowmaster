from app import st
import streamlit.web.bootstrap as bootstrap

def handler(event, context):
    return bootstrap.run(st, '', '', '')

if __name__ == '__main__':
    bootstrap.run(st, '', '', '')
