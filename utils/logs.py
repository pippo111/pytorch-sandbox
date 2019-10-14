def to_table(html_table, filename):
    header = ''
    footer = ''

    with open(f'utils/html/header.html', 'r') as header_html:
        header = header_html.read()

    with open(f'utils/html/footer.html', 'r') as footer_html:
        footer = footer_html.read()

    with open(filename, 'w') as html_file:
        html_file.write(header)
        html_file.write(html_table)
        html_file.write(footer)
