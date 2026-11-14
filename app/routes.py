
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from sqlalchemy.exc import SQLAlchemyError

from . import db
from .models import Sale

from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio

from ml.forecasting import train_forecast_model


main = Blueprint("main", __name__)


# --------------------------------------------------
# HELPER: CONVERT PLOTLY CHART TO HTML
# --------------------------------------------------

def chart_to_html(fig):
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "responsive": True,
            "displaylogo": False
        }
    )


# --------------------------------------------------
# DASHBOARD ROUTE
# --------------------------------------------------

@main.route("/", methods=["GET"])
def dashboard():

    # Get all sales for forecasting.
    # Forecasting should not change when dashboard
    # filters are applied.
    all_sales = Sale.query.order_by(
        Sale.sale_date.asc()
    ).all()

    # Get filter values
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    selected_region = request.args.get("region", "")
    selected_category = request.args.get("category", "")

    # Main dashboard query
    query = Sale.query

    # Date filters
    if start_date:
        try:
            start = datetime.strptime(
                start_date, "%Y-%m-%d"
            ).date()

            query = query.filter(
                Sale.sale_date >= start
            )
        except ValueError:
            flash("Invalid start date.")

    if end_date:
        try:
            end = datetime.strptime(
                end_date, "%Y-%m-%d"
            ).date()

            query = query.filter(
                Sale.sale_date <= end
            )
        except ValueError:
            flash("Invalid end date.")

    # Region filter
    if selected_region:
        query = query.filter(
            Sale.region == selected_region
        )

    # Category filter
    if selected_category:
        query = query.filter(
            Sale.category == selected_category
        )

    sales = query.order_by(
        Sale.sale_date.asc()
    ).all()

    # --------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------

    revenue = sum(
        sale.revenue for sale in sales
    )

    profit = sum(
        sale.profit for sale in sales
    )

    orders = len(sales)

    products = sum(
        sale.quantity for sale in sales
    )

    profit_margin = (
        (profit / revenue) * 100
        if revenue > 0 else 0
    )

    # Best-selling product based on quantity
    product_sales = {}

    for sale in sales:
        product_sales[sale.product] = (
            product_sales.get(sale.product, 0)
            + sale.quantity
        )

    best_product = (
        max(
            product_sales,
            key=product_sales.get
        )
        if product_sales else "N/A"
    )

    # --------------------------------------------------
    # FILTER DROPDOWN OPTIONS
    # --------------------------------------------------

    regions = [
        row[0]
        for row in db.session.query(
            Sale.region
        ).distinct().order_by(
            Sale.region
        ).all()
        if row[0]
    ]

    categories = [
        row[0]
        for row in db.session.query(
            Sale.category
        ).distinct().order_by(
            Sale.category
        ).all()
        if row[0]
    ]

    # --------------------------------------------------
    # PREPARE DATAFRAME
    # --------------------------------------------------

    charts = []

    if sales:

        df = pd.DataFrame([
            {
                "Date": sale.sale_date,
                "Product": sale.product,
                "Category": sale.category,
                "Region": sale.region,
                "Quantity": sale.quantity,
                "Revenue": sale.revenue,
                "Profit": sale.profit
            }
            for sale in sales
        ])

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df["Month"] = (
            df["Date"].dt.to_period("M").astype(str)
        )

        # --------------------------------------------------
        # CHART 1: MONTHLY REVENUE
        # --------------------------------------------------

        monthly_sales = (
            df.groupby("Month")["Revenue"]
            .sum()
            .reset_index()
        )

        fig1 = px.line(
            monthly_sales,
            x="Month",
            y="Revenue",
            markers=True,
            title="Monthly Revenue Trend"
        )

        fig1.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Revenue"
        )

        # --------------------------------------------------
        # CHART 2: TOP SELLING PRODUCTS
        # --------------------------------------------------

        top_products = (
            df.groupby("Product")["Quantity"]
            .sum()
            .reset_index()
            .sort_values(
                "Quantity",
                ascending=False
            )
            .head(10)
        )

        fig2 = px.bar(
            top_products,
            x="Product",
            y="Quantity",
            title="Top Selling Products",
            color="Quantity"
        )

        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Product",
            yaxis_title="Quantity Sold"
        )

        # --------------------------------------------------
        # CHART 3: REGIONAL SALES
        # --------------------------------------------------

        regional_sales = (
            df.groupby("Region")["Revenue"]
            .sum()
            .reset_index()
        )

        fig3 = px.pie(
            regional_sales,
            names="Region",
            values="Revenue",
            title="Revenue by Region",
            hole=0.4
        )

        fig3.update_layout(
            template="plotly_white"
        )

        # --------------------------------------------------
        # CHART 4: CATEGORY REVENUE AND PROFIT
        # --------------------------------------------------

        category_sales = (
            df.groupby("Category")[
                ["Revenue", "Profit"]
            ]
            .sum()
            .reset_index()
        )

        category_long = category_sales.melt(
            id_vars="Category",
            value_vars=["Revenue", "Profit"],
            var_name="Metric",
            value_name="Amount"
        )

        fig4 = px.bar(
            category_long,
            x="Category",
            y="Amount",
            color="Metric",
            barmode="group",
            title="Revenue and Profit by Category"
        )

        fig4.update_layout(
            template="plotly_white",
            xaxis_title="Category",
            yaxis_title="Amount"
        )

        # --------------------------------------------------
        # CHART 5: PRODUCT PROFITABILITY
        # --------------------------------------------------

        product_profit = (
            df.groupby("Product")["Profit"]
            .sum()
            .reset_index()
            .sort_values(
                "Profit",
                ascending=False
            )
            .head(10)
        )

        fig5 = px.bar(
            product_profit,
            x="Product",
            y="Profit",
            title="Product Profitability",
            color="Profit"
        )

        fig5.update_layout(
            template="plotly_white",
            xaxis_title="Product",
            yaxis_title="Profit"
        )

        # --------------------------------------------------
        # CHART 6: CATEGORY PERFORMANCE
        # --------------------------------------------------

        category_quantity = (
            df.groupby("Category")["Quantity"]
            .sum()
            .reset_index()
        )

        fig6 = px.bar(
            category_quantity,
            x="Category",
            y="Quantity",
            title="Category-wise Sales Performance",
            color="Quantity"
        )

        fig6.update_layout(
            template="plotly_white",
            xaxis_title="Category",
            yaxis_title="Quantity Sold"
        )

        # Keep this order because the template
        # displays charts[:4] and charts[4:].
        charts = [
            chart_to_html(fig1),
            chart_to_html(fig2),
            chart_to_html(fig3),
            chart_to_html(fig4),
            chart_to_html(fig5),
            chart_to_html(fig6)
        ]

    # --------------------------------------------------
    # MACHINE LEARNING FORECASTING
    # --------------------------------------------------

    forecast_result = None

    if all_sales:
        try:
            forecast_result = train_forecast_model(
                all_sales,
                forecast_periods=3
            )
        except Exception as error:
            print(
                "Forecasting error:",
                str(error)
            )

            forecast_result = {
                "success": False,
                "message": (
                    "Unable to generate forecast. "
                    "Please check the sales data and "
                    "forecasting module."
                )
            }

    # --------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------

    return render_template(
        "dashboard.html",

        revenue=revenue,
        profit=profit,
        orders=orders,
        products=products,
        best_product=best_product,
        profit_margin=profit_margin,

        charts=charts,

        regions=regions,
        categories=categories,

        start_date=start_date,
        end_date=end_date,

        selected_region=selected_region,
        selected_category=selected_category,

        forecast_result=forecast_result
    )


# --------------------------------------------------
# CSV UPLOAD ROUTE
# --------------------------------------------------

@main.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if not file or file.filename == "":
        flash("Please select a CSV file.")
        return redirect(url_for("main.dashboard"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.")
        return redirect(url_for("main.dashboard"))

    required_columns = [
        "invoice_no",
        "product",
        "category",
        "quantity",
        "unit_price",
        "cost_price",
        "sale_date",
        "region"
    ]

    try:
        df = pd.read_csv(file)

        # Normalize column names
        df.columns = (
            df.columns.str.strip().str.lower()
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            flash(
                "Missing CSV columns: "
                + ", ".join(missing_columns)
            )
            return redirect(
                url_for("main.dashboard")
            )

        if df.empty:
            flash("The uploaded CSV file is empty.")
            return redirect(
                url_for("main.dashboard")
            )

        # Validate and clean data
        df = df[required_columns].copy()

        df["invoice_no"] = (
            df["invoice_no"].astype(str).str.strip()
        )

        for column in [
            "product",
            "category",
            "region"
        ]:
            df[column] = (
                df[column].astype(str).str.strip()
            )

        df["quantity"] = pd.to_numeric(
            df["quantity"],
            errors="raise"
        )

        df["unit_price"] = pd.to_numeric(
            df["unit_price"],
            errors="raise"
        )

        df["cost_price"] = pd.to_numeric(
            df["cost_price"],
            errors="raise"
        )

        df["sale_date"] = pd.to_datetime(
            df["sale_date"],
            errors="raise"
        ).dt.date

        if (
            (df["quantity"] <= 0).any()
            or (df["unit_price"] < 0).any()
            or (df["cost_price"] < 0).any()
        ):
            flash(
                "Quantity must be positive and "
                "prices cannot be negative."
            )
            return redirect(
                url_for("main.dashboard")
            )

        if df[required_columns].isnull().any().any():
            flash(
                "The CSV contains missing values. "
                "Please correct the file."
            )
            return redirect(
                url_for("main.dashboard")
            )

        # Avoid duplicate invoice numbers
        existing_invoices = {
            row[0]
            for row in db.session.query(
                Sale.invoice_no
            ).all()
        }

        new_records = []

        for _, row in df.iterrows():

            invoice = row["invoice_no"]

            if invoice in existing_invoices:
                continue

            new_records.append(
                Sale(
                    invoice_no=invoice,
                    product=row["product"],
                    category=row["category"],
                    quantity=int(row["quantity"]),
                    unit_price=float(row["unit_price"]),
                    cost_price=float(row["cost_price"]),
                    sale_date=row["sale_date"],
                    region=row["region"]
                )
            )

            existing_invoices.add(invoice)

        if not new_records:
            flash(
                "No new records uploaded. "
                "All invoice numbers already exist."
            )
            return redirect(
                url_for("main.dashboard")
            )

        db.session.add_all(new_records)
        db.session.commit()

        flash(
            f"Successfully uploaded "
            f"{len(new_records)} new sales records."
        )

    except (ValueError, TypeError) as error:
        db.session.rollback()

        flash(
            "Invalid CSV data. Please check "
            "the dates and numeric columns."
        )

        print("CSV validation error:", str(error))

    except SQLAlchemyError as error:
        db.session.rollback()

        flash(
            "Database error occurred while "
            "uploading the CSV."
        )

        print("Database error:", str(error))

    except Exception as error:
        db.session.rollback()

        flash("Unable to upload the CSV file.")

        print("Upload error:", str(error))

    return redirect(
        url_for("main.dashboard")
    )
