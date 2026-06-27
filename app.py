from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_sale_price_study import page_sale_price_study_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_sale_price import page_predict_sale_price_body


app = MultiPage(app_name="Heritage Housing Issues")

app.add_page("Project Summary", page_summary_body)
app.add_page("Sale Price Study", page_sale_price_study_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("Predict Sale Price", page_predict_sale_price_body)

app.run()