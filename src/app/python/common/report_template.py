"""
filename: report_template.py
Author: Prashant Verma
email: 
version:
issue_date:
update_date:
"""
from jinja2 import Template
from xhtml2pdf import pisa
from io import BytesIO
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.common.config_manager import cfg
import os

class ReportTemplate:
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
        
            }
            .header {
                background-image: url("{{ header_image }}");
                background-size: contain;
                background-repeat: no-repeat;
                height: 150px; /* Adjust as needed */
                width: 100%;
                text-align: center;
                color: white;
                padding: 20px;
            }
            .footer {
                position: fixed;
                bottom: 0;
                width: 100%;
                background-image: url("{{ footer_image }}");
                background-size: contain;
                background-repeat: no-repeat;
                height: 50px; 
                width: 100%;
                text-align: center;
                color: white;
                padding: 20px;
            }
            .content {
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header"></div>
        <div>
            <h1>{% if header_text %}{{ header_text }}{% else %}SABIC Report{% endif %}</h1>
        </div>
        <div class="content">
            {% for section in sections %}
            <h2>{{ section.heading }}</h2>
            <p>{{ section.content }}</p>
            {% endfor %}
                     
        </div>
        <div class="footer">
            <p>sabic</p>
        </div>
    </body>
    </html>
    """

    def __init__(self, response):
        self.response = response
        # self.table_status = status
        # self.json_table = json_tbl

    def render_template(self):
        header_image_path = str(constant.PROJECT_ROOT_DIR)+str(cfg.get_resource_config(constant.HEADER_IMAGE_FILE))
        footer_image_path = str(constant.PROJECT_ROOT_DIR)+str(cfg.get_resource_config(constant.FOOTER_IMAGE_FILE))
        sections = []
        response_text = self.response['output_text']
        for section in response_text.split("\n\n"):
            parts = section.split("\n\n", 1)
            if len(parts) == 2:
                heading, content = parts
                sections.append({"heading": heading.strip("** "), "content": content})
            else:
                heading = section.strip("** ")
                sections.append({"heading": heading, "content": ""})

        template = Template(self.template_str)
        rendered_template = template.render(
            header_image= header_image_path,
            footer_image= footer_image_path,
            header_text="SABIC - Report",
            sections=sections,
           
        )
        return rendered_template


