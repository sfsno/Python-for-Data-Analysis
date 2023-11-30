from django.urls import path

from . import views

urlpatterns = [
    path("", views.index0, name="index0"),
    #path("", views.index5, name="index5"),
    path("apply_criteria", views.index1, name="index1"),
    path("apply_personnality", views.index2, name="index2"),
    path("apply_corr", views.index3, name="index3"),
    path("apply_roc", views.index4, name="index4"),
    #path("", views.index4, name="index4"),
]