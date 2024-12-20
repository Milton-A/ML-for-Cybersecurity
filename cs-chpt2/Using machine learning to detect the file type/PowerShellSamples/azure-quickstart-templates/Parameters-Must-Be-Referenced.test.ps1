﻿param(
[Parameter(Mandatory=$true,Position=0)]
[PSObject]
$TemplateObject
)
foreach ($parameter in $TemplateObject.parameters.psobject.properties) {
    if ($TemplateText -notmatch "parameters\(['`"]$($Parameter.Name)['`"]\)") {
        Write-Error -Message "Unreferenced parameter: $($Parameter.Name)" `
            -ErrorId Parameters.Must.Be.Referenced -TargetObject $parameter
    }
}
 



