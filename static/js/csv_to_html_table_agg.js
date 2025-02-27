var CsvToHtmlTableAgg = CsvToHtmlTableAgg || {};

CsvToHtmlTableAgg = {
    init: function (options) {
        options = options || {};
        var csv_path = options.csv_path || "";
        var el = options.element || "table-container";
        var csv_options = options.csv_options || {};
        var datatables_options = options.datatables_options || {};
        var custom_formatting = options.custom_formatting || [];
        var customTemplates = {};
        $.each(custom_formatting, function (i, v) {
            var colIdx = v[0];
            var func = v[1];
            customTemplates[colIdx] = func;
        });

        var $table = $("<table class='table table-striped table-condensed' id='" + el + "-table'></table>");
        var $containerElement = $("#" + el);
        $containerElement.empty().append($table);

        $.when($.get(csv_path)).then(
            function (data) {
                var csvData = $.csv.toArrays(data, csv_options);
                var $tableHead = $("<thead></thead>");
                var csvHeaderRow = csvData[0];

                var $tableHeadRow1 = $("<tr></tr>");
                $tableHeadRow1.append($("<th colspan='3' data-dt-order='disable' class='has-text-centered'></th>").text(""));
                $tableHeadRow1.append($("<th colspan='8' data-dt-order='disable' class='has-text-centered'></th>").text("Average Performance Across Seven Tasks"));
                $tableHeadRow1.css("background-color", "#f5f5f5");
                $tableHead.append($tableHeadRow1);

                var $tableHeadRow2 = $("<tr></tr>");

                for (var headerIdx = 0; headerIdx < csvHeaderRow.length; headerIdx++) {
                    if (headerIdx == 3) {
                        // skip the third column
                        continue;
                    }
                    $tableHeadRow2Cell = $("<th class='tooltip'></th>").text(csvHeaderRow[headerIdx]);
                    $tableHeadRow2.append($tableHeadRow2Cell);
                }
                $tableHeadRow2.css("background-color", "#f5f5f5");
                $tableHead.append($tableHeadRow2);

                $table.append($tableHead);
                var $tableBody = $("<tbody></tbody>");

                for (var rowIdx = 1; rowIdx < csvData.length; rowIdx++) {
                    var $tableBodyRow = $("<tr></tr>");
                    for (var colIdx = 0; colIdx < csvData[rowIdx].length; colIdx++) {
                        if (colIdx == 3) {
                            // skip the third column
                            continue;
                        }
                        var $tableBodyRowTd = $("<td></td>");
                        var cellTemplateFunc = customTemplates[colIdx];
                        if (cellTemplateFunc) {
                            $tableBodyRowTd.html(cellTemplateFunc(csvData[rowIdx][colIdx]));
                        } else {
                            $tableBodyRowTd.text(csvData[rowIdx][colIdx]);
                        }
                        if (colIdx == 0 ||  colIdx == 2 || colIdx == 10 || colIdx == 13 || colIdx == 16 ) {
                            $tableBodyRowTd.css("border-right", "1px solid #dbdbdb");
                        }
                        // // if the second column equals to "Proprietary", then set the background color of the row to light red
                        // if (colIdx == 1 && csvData[rowIdx][colIdx] == "Proprietary") {
                        //     $tableBodyRow.css("background-color", "#FEFAE3");
                        // }
                        // if the second column equals to "Open", then set the background color of the row to light green
                        // if (colIdx == 1 && csvData[rowIdx][colIdx] == "Open") {
                        //     $tableBodyRow.css("background-color", "#F8FBFD");
                        // }
                        // if the second column equals to "Domain-specific", then set the background color of the row to light yellow
                        // if (colIdx == 1 && csvData[rowIdx][colIdx] == "Domain-specific") {
                        //     $tableBodyRow.css("background-color", "#ECFFE6");
                        // }
                        // if N/A, light blue

                        if (csvData[rowIdx][3] == "♭") {
                            $tableBodyRow.css("background-color", "#FEF4E4");
                        }
                        if (csvData[rowIdx][3] == "♯") {
                            $tableBodyRow.css("background-color", "#e3eff6");
                        }

                        // right align the second and third columns
                        if (colIdx == 1 || colIdx == 2) {
                            $tableBodyRowTd.css("text-align", "right");
                        }

                        if (csvData[rowIdx][colIdx] == "N/A") {
                            $tableBodyRow.css("background-color", "#FEF4E4");
                        }

                        $tableBodyRow.append($tableBodyRowTd);
                        $tableBody.append($tableBodyRow);
                    }
                }
                $table.append($tableBody);
                $table.DataTable(datatables_options);
            });
    }
};