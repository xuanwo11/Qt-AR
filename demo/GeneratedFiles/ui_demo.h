/********************************************************************************
** Form generated from reading UI file 'demo.ui'
**
** Created by: Qt User Interface Compiler version 5.3.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DEMO_H
#define UI_DEMO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_demoClass
{
public:

    void setupUi(QWidget *demoClass)
    {
        if (demoClass->objectName().isEmpty())
            demoClass->setObjectName(QStringLiteral("demoClass"));
        demoClass->resize(600, 400);

        retranslateUi(demoClass);

        QMetaObject::connectSlotsByName(demoClass);
    } // setupUi

    void retranslateUi(QWidget *demoClass)
    {
        demoClass->setWindowTitle(QApplication::translate("demoClass", "demo", 0));
    } // retranslateUi

};

namespace Ui {
    class demoClass: public Ui_demoClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DEMO_H
