
from app import db

class Sale(db.Model):
    __tablename__ = "sales"

    id = db.Column(db.Integer, primary_key=True)
    invoice_no = db.Column(db.String(30))
    product = db.Column(db.String(150))
    category = db.Column(db.String(100))
    quantity = db.Column(db.Integer)
    unit_price = db.Column(db.Float)
    cost_price = db.Column(db.Float, default=0)
    sale_date = db.Column(db.Date)
    region = db.Column(db.String(100))

    @property
    def revenue(self):
        return self.quantity * self.unit_price

    @property
    def profit(self):
        return self.quantity * (
            self.unit_price - self.cost_price
        )
