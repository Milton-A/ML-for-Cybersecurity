Next, select a file you would like to check your rules against. Call it
target_file. In a terminal, execute Yara rules.yara target_file as
follows:
    
    Yara rule.yara PythonBrochure

#The result should be as follows:
    
    is_a_pdf target_file
    dummy_rule2 target_rule




### in Step 1, we copied several YARA rules. The first rule checks the
magic numbers of a file to see if they match those of a PDF. The other two rules are trivial
rules—one that matches every file, and one that matches no file. Then, in Step 2, we used
the YARA program to run the rules against the target file. We saw from a printout that the
file matched some rules but not others, as expected from an effective YARA ruleset.
