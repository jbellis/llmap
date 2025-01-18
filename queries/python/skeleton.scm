; Class declarations
(class_definition
  name: (identifier) @name
  body: (block)) @class.declaration

; Function/method declarations
(function_definition
  name: (identifier) @name
  body: (block)) @method.declaration

; Instance variable declarations
(assignment 
  left: (attribute
    object: (identifier) @obj
    (#eq? @obj "self"))) @field.declaration
