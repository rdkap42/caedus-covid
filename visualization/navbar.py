import dash_bootstrap_components as dbc

def Navbar():         
    
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("Model", href="#")),
            dbc.NavItem(dbc.NavLink("News", href="#")),
            dbc.NavItem(dbc.NavLink("About", href="#")),
            dbc.NavItem(dbc.NavLink("Partners", href="#")),
            dbc.NavItem(dbc.NavLink("Contact", href="#"))
        ],
        brand="Caedus Covid",
        brand_href="#",
        color="primary",
        dark=True,
        style={'text-align': 'left'}
    )

    return navbar