out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
in float T;

uniform vec3 lightPos; 
uniform vec3 viewPos; 
uniform vec3 lightColor;

void main()
{
  vec4 objectColor = colormap(T, 33, 13, 10);
  // ambient
  float ambientStrength = 0.25;
  vec3 ambient = ambientStrength * lightColor;
  	
  // diffuse 
  vec3 norm = normalize(Normal);
  vec3 lightDir = normalize(lightPos - FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor;
    
  // specular
  float specularStrength = 0.7;
  vec3 viewDir = normalize(viewPos - FragPos);
  vec3 reflectDir = reflect(-lightDir, norm);  
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
  vec3 specular = specularStrength * spec * lightColor;  
        
  vec3 result = (ambient + diffuse + specular) * objectColor.xyz;
  FragColor = vec4(result, objectColor.a);
  /* if (T > 0.5) */
    /* FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f); */
  /* FragColor = objectColor; */
} 
