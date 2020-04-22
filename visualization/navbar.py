import dash_bootstrap_components as dbc

def Navbar():         
    
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("About", href="#")),
            dbc.NavItem(dbc.NavLink("Test", href="#"))
        ],
        brand="Caedus Covid",
        brand_href="#",
        color="primary",
        dark=True,
        style={'text-align': 'left'}
    )

    return navbar