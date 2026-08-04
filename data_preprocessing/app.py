from flask import Flask
 
from routes.checkpoint import bp as checkpoint_bp
from routes.folders    import bp as folders_bp
from routes.parser     import bp as parser_bp
from routes.references import bp as references_bp
from routes.rml        import bp as rml_bp
from routes.context_route        import bp as context_route_bp
from routes.stats_route import bp as stats_bp
def create_app() -> Flask:
    app = Flask(__name__)
 
    app.register_blueprint(folders_bp)
    app.register_blueprint(references_bp)
    app.register_blueprint(checkpoint_bp)
    app.register_blueprint(parser_bp)
    app.register_blueprint(rml_bp)
    app.register_blueprint(context_route_bp)
    app.register_blueprint(stats_bp)


    return app
 
 
if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=3000, debug=True)