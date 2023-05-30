

def get_table_as_svg(
        df,
        path,
        figwidth=400,
        figheight=400,
        ):
    
    ncols = len(df.columns)
    colwidth = figwidth/ncols
    rowheight = figheight/ (len(df)+1) #plus one for the header row. 

    def gettxt(st, x, y, length, height, va='central', cl='normal'):
        
        """Adds a text element. x and y refer to the bottom left of the cell. 
        The actual x,y position of the text is inferred from the cell 
        type (i.e. is it a column header or not, and is it a string or number) """
        
        _y = y - height/2 #y alignment is always the same
        
        if cl == 'heavy': #it's a header cell
            _x = x + length/2
            ha = 'middle'
            
        else: #it's a value cell.
            if isinstance(st, str): #strings go in the middle of the cell
                _x = x + length/2
                ha = 'middle'
            else: #its a float.
                _x = x+length - length/10
                ha = 'end'
        
        ln = f"  <text x=\"{_x}\" y=\"{_y}\" text-anchor=\"{ha}\" class=\"{cl}\">{st}</text>  \n"
        return ln

    def hline(x1,y1, x2, y2):
        vln = f"  <line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" style=\"stroke:rgb(0,0,0);stroke-width:0.5\" />"
        return vln

    def hbox(nrow, rowheight, figwidth, ):
        hbox = f"""  <rect x="0" y="{(nrow+1)*rowheight}" width="{figwidth}" height="{rowheight}" style="fill:#eee;fill-opacity:0.8;stroke-width:0;stroke:rgb(0,0,0)" />\n"""
        return hbox
    
    f = open('{path}.svg', 'w', encoding='utf-8')

    f.write(f"""<svg version="1.1"
        baseProfile="full"
        width="{figwidth}" height="{figheight}"
        xmlns="http://www.w3.org/2000/svg">
        <style>
        .normal {{ font: normal 12px sans-serif; fill: black; dominant-baseline: central; }}
        .heavy {{ font: bold 12px sans-serif; fill: black; dominant-baseline: central; }}
        </style>
    
    """)


    for count, col in enumerate(df.columns):
        f.write( gettxt(col, count*colwidth, rowheight,colwidth, rowheight, cl='heavy') )
    
    f.write( hline(0, rowheight, figwidth, rowheight) )
    f.write('\n')

    shaded = True
    for rownum in range(df.shape[0]):
        row = df.iloc[rownum]
        if shaded:
            f.write(hbox(rownum, rowheight, figwidth))
        shaded = not shaded
        for count, value in enumerate(row):
            f.write( gettxt(value, (count)*colwidth, (rownum+2)*rowheight, colwidth, rowheight, cl='normal') )

    f.write(""" </svg>""")
    f.close()